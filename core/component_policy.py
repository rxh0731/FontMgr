"""自动优化各层共享的连通域保护数量策略。"""

# 主体簇用于估算空间中心，过多外围部件会把中心拉向污染区。
PRIMARY_CLUSTER_COMPONENT_LIMIT = 5

# 结构保护需要覆盖多部件字形及离散点画，允许保留更多主要连通域。
STRUCTURE_PROTECTION_COMPONENT_LIMIT = 8
