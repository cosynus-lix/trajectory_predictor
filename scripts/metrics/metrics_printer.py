import metricsgenerator

for s in range (4):
    for m in range (2):
        metric = metricsgenerator.metrics(m+1,s+1,1000)
        Delatpmse = metric.DeltapMSE()
        deltamse = metric.deltaMSE()
        standardMSE = metric.standardMSE()
        

