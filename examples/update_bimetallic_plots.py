from gaspy_regress.analysis import create_gridplot


# CO2RR
adsorbate = 'CO'
target = -0.67
bandwidth = 0.2
targets = (target-bandwidth, target+bandwidth)
filename = 'CO2RR/CO bimetallic map'
create_gridplot(adsorbate, targets, filename)


# HER
adsorbate = 'H'
target = -0.27
bandwidth = 0.2
targets = (target-bandwidth, target+bandwidth)
filename = 'HER/H bimetallic map'
create_gridplot(adsorbate, targets, filename)


# NH3 decomposition
adsorbate = 'N'
target = -0.91 / 2      # At 1% ammonia concentration
# target = -0.58 / 2        # At 50% ammonia concentration
bandwidth = 0.2
targets = (target-bandwidth, target+bandwidth)
filename = 'NH3_decomp/N bimetallic map'
create_gridplot(adsorbate, targets, filename)


# ORR (O)
adsorbate = 'O'
target = 2.49
bandwidth = 0.2
targets = (target-bandwidth, target+bandwidth)
filename = 'ORR/O bimetallic map'
create_gridplot(adsorbate, targets, filename)

# ORR (OH)
adsorbate = 'OH'
target = 0.923
bandwidth = 0.2
targets = (target-bandwidth, target+bandwidth)
filename = 'ORR/OH bimetallic map'
create_gridplot(adsorbate, targets, filename)

# ORR (OOH)
adsorbate = 'OOH'
target = 3.69
bandwidth = 0.2
targets = (target-bandwidth, target+bandwidth)
filename = 'ORR/OOH bimetallic map'
create_gridplot(adsorbate, targets, filename)
