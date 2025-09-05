module SingleIterationModule

using ADTypes: AutoEnzyme
using DynamicExpressions: AbstractExpression, string_tree, simplify_tree!, combine_operators
using ..UtilsModule: @threads_if
using ..CoreModule: AbstractOptions, Dataset, RecordType, create_expression, batch
using ..ComplexityModule: compute_complexity
using ..PopMemberModule: generate_reference
using ..PopulationModule: Population, finalize_costs
using ..HallOfFameModule: HallOfFame
using ..AdaptiveParsimonyModule: RunningSearchStatistics
using ..RegularizedEvolutionModule: reg_evol_cycle
using ..LossFunctionsModule: eval_cost
using ..ConstantOptimizationModule: optimize_constants
using ..LinearOptimizationModule: optimize_multilinear_individual
using ..RecorderModule: @recorder

# Cycle through regularized evolution many times,
# printing the fittest equation every 10% through
function s_r_cycle(
    dataset::D,
    pop::P,
    ncycles::Int,
    curmaxsize::Int,
    running_search_statistics::RunningSearchStatistics;
    verbosity::Int=0,
    options::AbstractOptions,
    record::RecordType,
)::Tuple{
    P,HallOfFame{T,L,N},Float64
} where {T,L,D<:Dataset{T,L},N<:AbstractExpression{T},P<:Population{T,L,N}}
    max_temp = 1.0
    min_temp = 0.0
    if !options.annealing
        min_temp = max_temp
    end
    all_temperatures = ncycles > 1 ? LinRange(max_temp, min_temp, ncycles) : [max_temp]
    best_examples_seen = HallOfFame(options, dataset)
    num_evals = 0.0

    batched_dataset = options.batching ? batch(dataset, options.batch_size) : dataset

    for temperature in all_temperatures
        pop, tmp_num_evals = reg_evol_cycle(
            batched_dataset,
            pop,
            temperature,
            curmaxsize,
            running_search_statistics,
            options,
            record,
        )
        num_evals += tmp_num_evals
        for member in pop.members
            size = compute_complexity(member, options)
            if 0 < size <= options.maxsize && (
                !best_examples_seen.exists[size] ||
                member.cost < best_examples_seen.members[size].cost
            )
                best_examples_seen.exists[size] = true
                best_examples_seen.members[size] = copy(member)
            end
        end
    end

    return (pop, best_examples_seen, num_evals)
end

function optimize_and_simplify_population(
    dataset::D, pop::P, options::AbstractOptions, curmaxsize::Int, record::RecordType;
    do_linear_optimization::Bool=true
)::Tuple{P,Float64} where {T,L,D<:Dataset{T,L},P<:Population{T,L}}
    # 创建一个数组用于存储每个个体的评估次数
    array_num_evals = zeros(Float64, pop.n)
    # 根据优化器概率随机决定哪些个体会进行常数优化
    do_optimization = rand(pop.n) .< options.optimizer_probability
    # 对于每个个体，以50%的概率决定是否进行线性优化
    do_linear_optimization = do_linear_optimization && (rand(pop.n) .< 0.5)  # 100% 概率进行线性优化
    # Note: we have to turn off this threading loop due to Enzyme, since we need
    # to manually allocate a new task with a larger stack for Enzyme.
    # 确定是否使用多线程：非确定性模式且不使用Enzyme自动微分时启用
    should_thread = !(options.deterministic) && !(isa(options.autodiff_backend, AutoEnzyme))

    # 如果启用批处理，则创建批处理数据集
    batched_dataset = options.batching ? batch(dataset, options.batch_size) : dataset

    # 根据should_thread决定是否使用多线程处理种群中的每个个体
    @threads_if should_thread for j in 1:(pop.n)
        # 如果启用了简化功能
        if options.should_simplify
            # 获取当前个体的表达式树
            tree = pop.members[j].tree
            # 简化表达式树
            tree = simplify_tree!(tree, options.operators)
            # 合并操作符（如合并连续的同类运算）
            tree = combine_operators(tree, options.operators)
            # 将简化后的树赋值回个体
            pop.members[j].tree = tree
        end
        # 如果启用了常数优化且当前个体需要优化
        if options.should_optimize_constants && do_optimization[j]
            # TODO: Might want to do full batch optimization here?
            # 对个体的常数进行优化，并记录评估次数
            pop.members[j], array_num_evals[j] = optimize_constants(
                batched_dataset, pop.members[j], options
            )
        end
        # 添加线性优化逻辑
        # 如果当前个体需要进行线性优化
        if do_linear_optimization[j]
            # 对个体进行多线性优化，并获取优化后的个体和评估次数
            optimized_member, linear_evals = optimize_multilinear_individual(
                pop.members[j], batched_dataset, options
            )
            # 更新个体为优化后的结果
            pop.members[j] = optimized_member
            # 累加线性优化的评估次数
            array_num_evals[j] += linear_evals
        end
    end
    # 计算总的评估次数
    num_evals = sum(array_num_evals)
    # 对种群的代价进行最终计算
    pop, tmp_num_evals = finalize_costs(dataset, pop, options)
    # 累加最终计算的评估次数
    num_evals += tmp_num_evals

    # Now, we create new references for every member,
    # and optionally record which operations occurred.
    # 为每个个体创建新的引用，并可选择记录发生的操作
    for j in 1:(pop.n)
        # 获取旧的引用ID
        old_ref = pop.members[j].ref
        # 生成新的引用ID
        new_ref = generate_reference()
        # 将父引用设置为旧的引用ID
        pop.members[j].parent = old_ref
        # 更新引用ID为新生成的ID
        pop.members[j].ref = new_ref

        # 使用@recorder宏记录操作（如果启用了记录功能）
        @recorder begin
            # Same structure as in RegularizedEvolution.jl,
            # except we assume that the record already exists.
            # 确保记录中存在mutations键
            @assert haskey(record, "mutations")
            # 获取当前个体
            member = pop.members[j]
            # 如果记录中不存在当前个体的记录，则创建新记录
            if !haskey(record["mutations"], "$(member.ref)")
                record["mutations"]["$(member.ref)"] = RecordType(
                    "events" => Vector{RecordType}(),
                    "tree" => string_tree(member.tree, options),
                    "cost" => member.cost,
                    "loss" => member.loss,
                    "parent" => member.parent,
                )
            end
            # 创建优化和简化事件记录
            optimize_and_simplify_event = RecordType(
                "type" => "tuning",
                "time" => time(),
                "child" => new_ref,
                "mutation" => RecordType(
                    "type" =>
                        # 根据是否进行了优化来确定事件类型
                        if (do_optimization[j] && options.should_optimize_constants)
                            "simplification_and_optimization"
                        else
                            "simplification"
                        end,
                ),
            )
            # 创建死亡事件记录（表示旧个体已被替换）
            death_event = RecordType("type" => "death", "time" => time())

            # 将优化和简化事件添加到旧个体的事件列表中
            push!(record["mutations"]["$(old_ref)"]["events"], optimize_and_simplify_event)
            # 将死亡事件添加到旧个体的事件列表中
            push!(record["mutations"]["$(old_ref)"]["events"], death_event)
        end
    end
    # 返回优化后的种群和评估次数
    return (pop, num_evals)
end

end
