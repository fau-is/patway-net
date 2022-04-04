Good parameters for simulation
- hidden seq fz size: 16
- epochs: 1000
- lr: 1e-4
- mlp hidden act: relu
- mlp hidden size: 16


Should be checked
- CRP values in one-hot encoding not used; also no outlier
- Early stopping with val_loss

Sequential Features
- Activity
- Leucocytes
- CRP
- LacticAcid
- org:group

Encoding
- Activity + CRP + LacticAcid + Leucocytes -> one-hot encoding
- org:group -> nicht betrachtet


Keras vs. pytorch?
- Early stopping -> val_loss; patience=10 (Keras) --> check
- Reduce LR on Plateau (Keras) --> remove it
- Bidirectionales LSTM (Keras) --> no simple LSTM



https://www.cs.toronto.edu/~lczhang/360/lec/w02/training.html

1) single forward not forget gate?
2) masking interaction / which feature?
3) interactions also for stat features? -> only one weight value

[x] check bceloss vs. bcelosswithlogit
[x] check optimizer.zero_grad()



-------- Sepsis / AIC / neurones per feature = 10
Epoch: 0 -- Loss: 1.8093899061789829
Epoch: 1 -- Loss: 1.7197494691400785
Epoch: 2 -- Loss: 1.641807693096886
Epoch: 3 -- Loss: 1.5698538044228334
Epoch: 4 -- Loss: 1.5041528020268486
Epoch: 5 -- Loss: 1.4404017181949746
Epoch: 6 -- Loss: 1.3827169989895522
Epoch: 7 -- Loss: 1.32790666184119
Epoch: 8 -- Loss: 1.277960211332932
Epoch: 9 -- Loss: 1.2317124646138653
Epoch: 10 -- Loss: 1.1886357163661772
Epoch: 11 -- Loss: 1.1543957246009358
Epoch: 12 -- Loss: 1.119969465339406
Epoch: 13 -- Loss: 1.0903864121004256
Epoch: 14 -- Loss: 1.0660682010719873
Epoch: 15 -- Loss: 1.0461406380932285
Epoch: 16 -- Loss: 1.0274797544793595
Epoch: 17 -- Loss: 1.0117858319977697
Epoch: 18 -- Loss: 0.9979549634349925
Epoch: 19 -- Loss: 0.9877843923399108
Epoch: 20 -- Loss: 0.9784340476254498
Epoch: 21 -- Loss: 0.9697844786466762
Epoch: 22 -- Loss: 0.9635867882969475
Epoch: 23 -- Loss: 0.9576907188913161
Epoch: 24 -- Loss: 0.9521403341021432
Epoch: 25 -- Loss: 0.9492346712062435
Epoch: 26 -- Loss: 0.9431549246238745
Epoch: 27 -- Loss: 0.9405236395014551
Epoch: 28 -- Loss: 0.9359522294569638
Epoch: 29 -- Loss: 0.9345461579409693
Epoch: 30 -- Loss: 0.931573318798521
Epoch: 31 -- Loss: 0.9290571159341193
Epoch: 32 -- Loss: 0.9265915313748402
Epoch: 33 -- Loss: 0.9280470326532206
Epoch: 34 -- Loss: 0.921832094328944
Epoch: 35 -- Loss: 0.921553586446217
Epoch: 36 -- Loss: 0.9210267866759648
Epoch: 37 -- Loss: 0.9198533825506853
Epoch: 38 -- Loss: 0.9163340112281456
Epoch: 39 -- Loss: 0.9179479363860086
Epoch: 40 -- Loss: 0.9137656990978357
Epoch: 41 -- Loss: 0.9133271947818208
Epoch: 42 -- Loss: 0.9132761266458046
Epoch: 43 -- Loss: 0.9108538856895297
Epoch: 44 -- Loss: 0.9104561274941313
Epoch: 45 -- Loss: 0.9100217234282847
Epoch: 46 -- Loss: 0.908489124581145
Epoch: 47 -- Loss: 0.9075748855353097
Epoch: 48 -- Loss: 0.9070655854826672
Epoch: 49 -- Loss: 0.9070646985186724
Epoch: 50 -- Loss: 0.9054614774348348
Epoch: 51 -- Loss: 0.9049093118009545
Epoch: 52 -- Loss: 0.9048235632135335
Epoch: 53 -- Loss: 0.9054414518977765
Epoch: 54 -- Loss: 0.9032262143500865
Epoch: 55 -- Loss: 0.9021914866253518
Epoch: 56 -- Loss: 0.9008247744889127
Epoch: 57 -- Loss: 0.9025040265854031
Epoch: 58 -- Loss: 0.9013552062395348
Epoch: 59 -- Loss: 0.901211355339651
Epoch: 60 -- Loss: 0.9002267847663308
Epoch: 61 -- Loss: 0.9009688214282469
Epoch: 62 -- Loss: 0.899882672227753
Epoch: 63 -- Loss: 0.8990757509589965
Epoch: 64 -- Loss: 0.8984953146373128
Epoch: 65 -- Loss: 0.9010516218701404
Epoch: 66 -- Loss: 0.8995707086089904
Epoch: 67 -- Loss: 0.8989304910240755
Epoch: 68 -- Loss: 0.8979286928935603
Epoch: 69 -- Loss: 0.8977584167787831
Epoch: 70 -- Loss: 0.8973703483966905
Epoch: 71 -- Loss: 0.8970506861805717
Epoch: 72 -- Loss: 0.8968580953715775
Epoch: 73 -- Loss: 0.8963783769501652
Epoch: 74 -- Loss: 0.8966438369577533
Epoch: 75 -- Loss: 0.8959306801095834
Epoch: 76 -- Loss: 0.8965004749659143
Epoch: 77 -- Loss: 0.8967424798378656
Epoch: 78 -- Loss: 0.8955736045331868
Epoch: 79 -- Loss: 0.8956246258918739
Epoch: 80 -- Loss: 0.8953590472151036
Epoch: 81 -- Loss: 0.8952635364478204
Epoch: 82 -- Loss: 0.8945109849057756
Epoch: 83 -- Loss: 0.8952692104990723
Epoch: 84 -- Loss: 0.894377876734809
Epoch: 85 -- Loss: 0.8935303755029791
Epoch: 86 -- Loss: 0.8950747943301376
Epoch: 87 -- Loss: 0.8946879800522866
Epoch: 88 -- Loss: 0.893884531048218
Epoch: 89 -- Loss: 0.8939240957822077
Epoch: 90 -- Loss: 0.8939236987795812
Epoch: 91 -- Loss: 0.8934977895701999
Epoch: 92 -- Loss: 0.8926766051912987
Epoch: 93 -- Loss: 0.8933176616663356
Epoch: 94 -- Loss: 0.8930616767944913
Epoch: 95 -- Loss: 0.8933970934017582
Epoch: 96 -- Loss: 0.8924591634272669
Epoch: 97 -- Loss: 0.8923105115483518
Epoch: 98 -- Loss: 0.8916770187281096
Epoch: 99 -- Loss: 0.8915503438772051
Degrees of freedom <= 0 for slice
invalid value encountered in double_scalars
auc
0,0.720997029156333
Degrees of freedom <= 0 for slice
invalid value encountered in double_scalars
Avg,0.720997029156333
Std,nan
precision (0)
0,0.8398887564560985
Avg,0.8398887564560985
Std,nan
precision (1)
0,0.7650602409638554
Avg,0.7650602409638554
Std,nan
recall (0)
0,0.9818857408267534
Avg,0.9818857408267534
Std,nan
recall (1)
0,0.23962264150943396
Avg,0.23962264150943396
Std,nan
f1-score (0)
0,0.9053533190578159
Avg,0.9053533190578159
Std,nan
f1-score (1)
0,0.36494252873563215
Avg,0.36494252873563215
Std,nan
support (0)
0,2153
Avg,2153.0
Std,nan
support (1)
0,530
Avg,530.0
Std,nan
accuracy
0,0.835259038389862
Avg,0.835259038389862
Std,nan