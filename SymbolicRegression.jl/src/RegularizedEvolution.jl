module RegularizedEvolutionModule

using DynamicExpressions: string_tree # 导入用于将表达式树转换为字符串的函数
using ..CoreModule: AbstractOptions, Dataset, RecordType, DATA_TYPE, LOSS_TYPE # 导入核心模块的抽象选项、数据集、记录类型、数据类型和损失类型
using ..PopulationModule: Population, best_of_sample # 导入种群模块的种群结构和从样本中选择最佳个体的函数
using ..AdaptiveParsimonyModule: RunningSearchStatistics # 导入自适应简约性模块的运行搜索统计信息
using ..MutateModule: next_generation, crossover_generation # 导入变异模块的下一代（变异）和交叉生成函数
using ..RecorderModule: @recorder # 导入记录器模块的宏，用于记录事件
using ..UtilsModule: argmin_fast # 导入工具模块的快速查找最小值的函数

# 循环遍历种群多次，用小样本中最适合的个体替换最老的个体
function reg_evol_cycle(
    dataset::Dataset{T,L}, # 数据集
    pop::P, # 种群
    temperature, # 温度（用于模拟退火）
    curmaxsize::Int, # 当前最大表达式大小
    running_search_statistics::RunningSearchStatistics, # 运行搜索统计信息
    options::AbstractOptions, # 选项
    record::RecordType, # 记录器
)::Tuple{P,Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:Population{T,L}} # 返回更新后的种群和评估次数
    num_evals = 0.0 # 初始化评估次数
    n_evol_cycles = ceil(Int, pop.n / options.tournament_selection_n) # 计算进化循环次数，基于种群大小和锦标赛选择数量

    for i in 1:n_evol_cycles # 遍历每个进化循环
        if rand() > options.crossover_probability # 如果随机数大于交叉概率，则进行变异
            allstar = best_of_sample(pop, running_search_statistics, options) # 从样本中选择最佳个体（allstar）
            mutation_recorder = RecordType() # 创建变异记录器
            baby, mutation_accepted, tmp_num_evals = next_generation( # 生成下一代（变异）
                dataset,
                allstar,
                temperature,
                curmaxsize,
                running_search_statistics,
                options;
                tmp_recorder=mutation_recorder, # 临时记录器
            )
            num_evals += tmp_num_evals # 累加评估次数

            if !mutation_accepted && options.skip_mutation_failures # 如果变异未被接受且跳过变异失败
                # 跳过此变异，而不是用未改变的成员替换最老的成员
                continue
            end

            oldest = argmin_fast([pop.members[member].birth for member in 1:(pop.n)]) # 找到种群中最老的个体索引

            @recorder begin # 记录变异事件
                if !haskey(record, "mutations") # 如果记录中没有“mutations”键
                    record["mutations"] = RecordType() # 创建“mutations”记录
                end
                for member in [allstar, baby, pop.members[oldest]] # 遍历allstar、新生成的个体和被替换的最老个体
                    if !haskey(record["mutations"], "$(member.ref)") # 如果记录中没有该成员的引用
                        record["mutations"]["$(member.ref)"] = RecordType( # 创建该成员的记录
                            "events" => Vector{RecordType}(), # 事件列表
                            "tree" => string_tree(member.tree, options), # 表达式树字符串
                            "cost" => member.cost, # 代价
                            "loss" => member.loss, # 损失
                            "parent" => member.parent, # 父代引用
                        )
                    end
                end
                mutate_event = RecordType( # 创建变异事件记录
                    "type" => "mutate", # 事件类型
                    "time" => time(), # 时间戳
                    "child" => baby.ref, # 子代引用
                    "mutation" => mutation_recorder, # 变异记录器
                )
                death_event = RecordType("type" => "death", "time" => time()) # 创建死亡事件记录

                # 使用随机键而不是向量；否则会有冲突！
                push!(record["mutations"]["$(allstar.ref)"]["events"], mutate_event) # 将变异事件添加到allstar的事件列表
                push!(
                    record["mutations"]["$(pop.members[oldest].ref)"]["events"], death_event # 将死亡事件添加到最老个体的事件列表
                )
            end

            pop.members[oldest] = baby # 用新生成的个体替换最老的个体

        else # Crossover (交叉)
            allstar1 = best_of_sample(pop, running_search_statistics, options) # 选择第一个最佳个体
            allstar2 = best_of_sample(pop, running_search_statistics, options) # 选择第二个最佳个体

            crossover_recorder = RecordType() # 创建交叉记录器
            baby1, baby2, crossover_accepted, tmp_num_evals = crossover_generation( # 生成两个新个体（交叉）
                allstar1,
                allstar2,
                dataset,
                curmaxsize,
                options;
                recorder=crossover_recorder, # 记录器
            )
            num_evals += tmp_num_evals # 累加评估次数

            if !crossover_accepted && options.skip_mutation_failures # 如果交叉未被接受且跳过变异失败
                continue
            end

            # 找到要替换的最老成员：
            oldest1 = argmin_fast([pop.members[member].birth for member in 1:(pop.n)]) # 找到第一个最老个体索引
            BT = typeof(first(pop.members).birth) # 获取出生时间类型
            oldest2 = argmin_fast([ # 找到第二个最老个体索引，排除第一个最老个体
                i == oldest1 ? typemax(BT) : pop.members[i].birth for i in 1:(pop.n)
            ])

            @recorder begin # 记录交叉事件
                if !haskey(record, "mutations") # 如果记录中没有“mutations”键
                    record["mutations"] = RecordType() # 创建“mutations”记录
                end
                for member in [ # 遍历所有相关成员
                    allstar1,
                    allstar2,
                    baby1,
                    baby2,
                    pop.members[oldest1],
                    pop.members[oldest2],
                ]
                    if !haskey(record["mutations"], "$(member.ref)") # 如果记录中没有该成员的引用
                        record["mutations"]["$(member.ref)"] = RecordType( # 创建该成员的记录
                            "events" => Vector{RecordType}(), # 事件列表
                            "tree" => string_tree(member.tree, options), # 表达式树字符串
                            "cost" => member.cost, # 代价
                            "loss" => member.loss, # 损失
                            "parent" => member.parent, # 父代引用
                        )
                    end
                end
                crossover_event = RecordType( # 创建交叉事件记录
                    "type" => "crossover", # 事件类型
                    "time" => time(), # 时间戳
                    "parent1" => allstar1.ref, # 父代1引用
                    "parent2" => allstar2.ref, # 父代2引用
                    "child1" => baby1.ref, # 子代1引用
                    "child2" => baby2.ref, # 子代2引用
                    "details" => crossover_recorder, # 交叉记录器
                )
                death_event1 = RecordType("type" => "death", "time" => time()) # 创建第一个死亡事件记录
                death_event2 = RecordType("type" => "death", "time" => time()) # 创建第二个死亡事件记录

                push!(record["mutations"]["$(allstar1.ref)"]["events"], crossover_event) # 将交叉事件添加到allstar1的事件列表
                push!(record["mutations"]["$(allstar2.ref)"]["events"], crossover_event) # 将交叉事件添加到allstar2的事件列表
                push!(
                    record["mutations"]["$(pop.members[oldest1].ref)"]["events"],
                    death_event1,
                ) # 将死亡事件添加到第一个最老个体的事件列表
                push!(
                    record["mutations"]["$(pop.members[oldest2].ref)"]["events"],
                    death_event2,
                ) # 将死亡事件添加到第二个最老个体的事件列表
            end

            # 用新成员替换旧成员：
            pop.members[oldest1] = baby1 # 用baby1替换oldest1
            pop.members[oldest2] = baby2 # 用baby2替换oldest2
        end
    end

    return (pop, num_evals) # 返回更新后的种群和评估次数
end

end
