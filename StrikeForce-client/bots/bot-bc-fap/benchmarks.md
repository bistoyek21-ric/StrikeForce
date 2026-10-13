# Benchmarks:

## pre-ewc+kd-backup [MASK1]:
**Experiment 1**:
```
================ METRICS ================
KILLS = 98, T_ALIVE = 8508
A_HP = 67030616, A_STAMINA = 85815360, A_DAMAGE = 7221419
=========================================
```
**Experiment 2**:
```
================ METRICS ================
KILLS = 50, T_ALIVE = 6846
A_HP = 99508145, A_STAMINA = 74507800, A_DAMAGE = 5947500
=========================================
```
**Experiemtn 3**:
```
================ METRICS ================
KILLS = 25, T_ALIVE = 4988
A_HP = 55995707, A_STAMINA = 42063440, A_DAMAGE = 3245441
=========================================
```
**Experiment 4**:
```
================ METRICS ================
KILLS = 127, T_ALIVE = 12600
A_HP = 151921786, A_STAMINA = 101329260, A_DAMAGE = 8931438
=========================================
```
**Experiment 5**:
```
================ METRICS ================
KILLS = 265, T_ALIVE = 16936
A_HP = 164437191, A_STAMINA = 139311080, A_DAMAGE = 20791165
=========================================
```
**Experiment 6**:
```
================ METRICS ================
KILLS = 11, T_ALIVE = 4796
A_HP = 54588711, A_STAMINA = 29008840, A_DAMAGE = 2106212
=========================================
```
**Experiment 7**:
```
================ METRICS ================
KILLS = 311, T_ALIVE = 20120
A_HP = 227526997, A_STAMINA = 167986560, A_DAMAGE = 26084835
=========================================
```
**Experiment 8**:
```
================ METRICS ================
KILLS = 115, T_ALIVE = 6992
A_HP = 78051994, A_STAMINA = 66891420, A_DAMAGE = 5028118
=========================================
```
**Experiment 9**:
```
================ METRICS ================
KILLS = 69, T_ALIVE = 10368
A_HP = 117155598, A_STAMINA = 73851420, A_DAMAGE = 5743037
=========================================
```
**Experiment 10**:
```
================ METRICS ================
KILLS = 264, T_ALIVE = 18176
A_HP = 142019041, A_STAMINA = 183535200, A_DAMAGE = 28372820
=========================================
```

------------------

## pre-ewc+kd-backup (head 0) [MASK1]:

**Experiment 1**:
```
================ METRICS ================
KILLS = 135, T_ALIVE = 15666
A_HP = 276536822, A_STAMINA = 120436080, A_DAMAGE = 8835834
=========================================
```
**Experiment 2**:
```
================ METRICS ================
KILLS = 22, T_ALIVE = 4710
A_HP = 68582621, A_STAMINA = 44715460, A_DAMAGE = 3055059
=========================================
```
**Experiment 3**:
```
================ METRICS ================
KILLS = 111, T_ALIVE = 9516
A_HP = 109750864, A_STAMINA = 94615000, A_DAMAGE = 9405375
=========================================
```
**Experiment 4**:
```
================ METRICS ================
KILLS = 16, T_ALIVE = 5376
A_HP = 60824128, A_STAMINA = 39455660, A_DAMAGE = 2596619
=========================================
```
**Experiment 5**:
```
================ METRICS ================
KILLS = 5, T_ALIVE = 4230
A_HP = 51403524, A_STAMINA = 26035340, A_DAMAGE = 1867183
=========================================
```
**Experiment 6**:
```
================ METRICS ================
KILLS = 101, T_ALIVE = 7733
A_HP = 118563860, A_STAMINA = 81717410, A_DAMAGE = 8229213
=========================================
```
**Experiment 7**:
```
================ METRICS ================
KILLS = 43, T_ALIVE = 6422
A_HP = 75531499, A_STAMINA = 65775660, A_DAMAGE = 4861756
=========================================
```
**Experiment 8**:
```
================ METRICS ================
KILLS = 61, T_ALIVE = 13550
A_HP = 213122205, A_STAMINA = 163621740, A_DAMAGE = 14816905
=========================================
```
**Experiment 9**:
```
================ METRICS ================
KILLS = 8, T_ALIVE = 4764
A_HP = 57086075, A_STAMINA = 30993580, A_DAMAGE = 2141428
=========================================
```
**Experiment 10**:
```
================ METRICS ================
KILLS = 37, T_ALIVE = 5652
A_HP = 65635168, A_STAMINA = 53082160, A_DAMAGE = 3187216
=========================================
```

------------------

## pre-ewc+kd-backup (linear aggregator) [MASK1]:

```
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 75, T_ALIVE = 5556
A_HP = 49230140, A_STAMINA = 63596160, A_DAMAGE = 5720386
A_HP/T_ALIVE = 8860.72, A_STAMINA/T_ALIVE = 11446.4, A_DAMAGE/T_ALIVE = 1029.59
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2777
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 195
Wall Collisions   (-): 56
Adjancent Enemies (-): 32
Flanking Enemies  (-): 107
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 896
Wall Collisions   (+): 522
Adjancent Enemies (+): 359
Flanking Enemies  (+): 15
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 122, T_ALIVE = 7832
A_HP = 54216830, A_STAMINA = 98334640, A_DAMAGE = 15309630
A_HP/T_ALIVE = 6922.48, A_STAMINA/T_ALIVE = 12555.5, A_DAMAGE/T_ALIVE = 1954.75
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3915
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 287
Wall Collisions   (-): 103
Adjancent Enemies (-): 9
Flanking Enemies  (-): 175
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1250
Wall Collisions   (+): 1121
Adjancent Enemies (+): 116
Flanking Enemies  (+): 13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 253, T_ALIVE = 16130
A_HP = 115007440, A_STAMINA = 170454220, A_DAMAGE = 44986330
A_HP/T_ALIVE = 7130.03, A_STAMINA/T_ALIVE = 10567.5, A_DAMAGE/T_ALIVE = 2788.99
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 8064
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 880
Wall Collisions   (-): 487
Adjancent Enemies (-): 17
Flanking Enemies  (-): 376
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2754
Wall Collisions   (+): 2497
Adjancent Enemies (+): 240
Flanking Enemies  (+): 17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 123, T_ALIVE = 7516
A_HP = 45282160, A_STAMINA = 95872520, A_DAMAGE = 10419390
A_HP/T_ALIVE = 6024.77, A_STAMINA/T_ALIVE = 12755.8, A_DAMAGE/T_ALIVE = 1386.29
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3757
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 397
Wall Collisions   (-): 204
Adjancent Enemies (-): 10
Flanking Enemies  (-): 183
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1141
Wall Collisions   (+): 1003
Adjancent Enemies (+): 129
Flanking Enemies  (+): 9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 56, T_ALIVE = 5014
A_HP = 38762880, A_STAMINA = 56122960, A_DAMAGE = 5024451
A_HP/T_ALIVE = 7730.93, A_STAMINA/T_ALIVE = 11193.3, A_DAMAGE/T_ALIVE = 1002.08
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2506
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 413
Wall Collisions   (-): 50
Adjancent Enemies (-): 283
Flanking Enemies  (-): 80
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 798
Wall Collisions   (+): 477
Adjancent Enemies (+): 309
Flanking Enemies  (+): 12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 312, T_ALIVE = 18222
A_HP = 199851215, A_STAMINA = 176082760, A_DAMAGE = 41730050
A_HP/T_ALIVE = 10967.6, A_STAMINA/T_ALIVE = 9663.2, A_DAMAGE/T_ALIVE = 2290.09
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 9110
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 889
Wall Collisions   (-): 444
Adjancent Enemies (-): 26
Flanking Enemies  (-): 419
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3079
Wall Collisions   (+): 2754
Adjancent Enemies (+): 296
Flanking Enemies  (+): 29
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 98, T_ALIVE = 6210
A_HP = 59899150, A_STAMINA = 80377540, A_DAMAGE = 12326406
A_HP/T_ALIVE = 9645.6, A_STAMINA/T_ALIVE = 12943.2, A_DAMAGE/T_ALIVE = 1984.93
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3104
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 283
Wall Collisions   (-): 113
Adjancent Enemies (-): 18
Flanking Enemies  (-): 152
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 970
Wall Collisions   (+): 785
Adjancent Enemies (+): 173
Flanking Enemies  (+): 12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 140, T_ALIVE = 7742
A_HP = 62900625, A_STAMINA = 95278220, A_DAMAGE = 10838670
A_HP/T_ALIVE = 8124.6, A_STAMINA/T_ALIVE = 12306.7, A_DAMAGE/T_ALIVE = 1399.98
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3870
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 321
Wall Collisions   (-): 101
Adjancent Enemies (-): 18
Flanking Enemies  (-): 202
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1068
Wall Collisions   (+): 913
Adjancent Enemies (+): 141
Flanking Enemies  (+): 14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 186, T_ALIVE = 10398
A_HP = 80729575, A_STAMINA = 119752780, A_DAMAGE = 24541185
A_HP/T_ALIVE = 7763.95, A_STAMINA/T_ALIVE = 11516.9, A_DAMAGE/T_ALIVE = 2360.18
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5198
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 605
Wall Collisions   (-): 346
Adjancent Enemies (-): 15
Flanking Enemies  (-): 244
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1577
Wall Collisions   (+): 1387
Adjancent Enemies (+): 176
Flanking Enemies  (+): 14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 202, T_ALIVE = 11582
A_HP = 113767475, A_STAMINA = 129018780, A_DAMAGE = 39193785
A_HP/T_ALIVE = 9822.78, A_STAMINA/T_ALIVE = 11139.6, A_DAMAGE/T_ALIVE = 3384.03
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5790
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 576
Wall Collisions   (-): 245
Adjancent Enemies (-): 22
Flanking Enemies  (-): 309
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1749
Wall Collisions   (+): 1547
Adjancent Enemies (+): 183
Flanking Enemies  (+): 19
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 231, T_ALIVE = 15012
A_HP = 109517070, A_STAMINA = 151911920, A_DAMAGE = 27757157
A_HP/T_ALIVE = 7295.3, A_STAMINA/T_ALIVE = 10119.4, A_DAMAGE/T_ALIVE = 1849
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7505
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 632
Wall Collisions   (-): 265
Adjancent Enemies (-): 57
Flanking Enemies  (-): 310
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2353
Wall Collisions   (+): 1873
Adjancent Enemies (+): 465
Flanking Enemies  (+): 15
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 110, T_ALIVE = 8132
A_HP = 77369725, A_STAMINA = 87191200, A_DAMAGE = 13774970
A_HP/T_ALIVE = 9514.23, A_STAMINA/T_ALIVE = 10722, A_DAMAGE/T_ALIVE = 1693.92
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4065
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 423
Wall Collisions   (-): 127
Adjancent Enemies (-): 94
Flanking Enemies  (-): 202
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1542
Wall Collisions   (+): 963
Adjancent Enemies (+): 562
Flanking Enemies  (+): 17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 127, T_ALIVE = 7454
A_HP = 78543720, A_STAMINA = 96473460, A_DAMAGE = 19800245
A_HP/T_ALIVE = 10537.1, A_STAMINA/T_ALIVE = 12942.5, A_DAMAGE/T_ALIVE = 2656.32
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3726
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 431
Wall Collisions   (-): 231
Adjancent Enemies (-): 14
Flanking Enemies  (-): 186
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1005
Wall Collisions   (+): 885
Adjancent Enemies (+): 116
Flanking Enemies  (+): 4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 426, T_ALIVE = 24510
A_HP = 210946080, A_STAMINA = 181258120, A_DAMAGE = 61126500
A_HP/T_ALIVE = 8606.53, A_STAMINA/T_ALIVE = 7395.27, A_DAMAGE/T_ALIVE = 2493.94
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 12254
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1332
Wall Collisions   (-): 740
Adjancent Enemies (-): 30
Flanking Enemies  (-): 562
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 4254
Wall Collisions   (+): 3790
Adjancent Enemies (+): 431
Flanking Enemies  (+): 33
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 148, T_ALIVE = 8362
A_HP = 52270150, A_STAMINA = 100032700, A_DAMAGE = 15703035
A_HP/T_ALIVE = 6250.91, A_STAMINA/T_ALIVE = 11962.8, A_DAMAGE/T_ALIVE = 1877.9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4180
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 329
Wall Collisions   (-): 104
Adjancent Enemies (-): 4
Flanking Enemies  (-): 221
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1213
Wall Collisions   (+): 1052
Adjancent Enemies (+): 150
Flanking Enemies  (+): 11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 113, T_ALIVE = 6948
A_HP = 54677515, A_STAMINA = 89900360, A_DAMAGE = 15472635
A_HP/T_ALIVE = 7869.53, A_STAMINA/T_ALIVE = 12939, A_DAMAGE/T_ALIVE = 2226.92
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3473
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 336
Wall Collisions   (-): 157
Adjancent Enemies (-): 11
Flanking Enemies  (-): 168
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 999
Wall Collisions   (+): 884
Adjancent Enemies (+): 104
Flanking Enemies  (+): 11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 108, T_ALIVE = 7672
A_HP = 68470475, A_STAMINA = 98142900, A_DAMAGE = 11920379
A_HP/T_ALIVE = 8924.72, A_STAMINA/T_ALIVE = 12792.3, A_DAMAGE/T_ALIVE = 1553.75
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3835
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 405
Wall Collisions   (-): 247
Adjancent Enemies (-): 18
Flanking Enemies  (-): 140
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1292
Wall Collisions   (+): 1084
Adjancent Enemies (+): 202
Flanking Enemies  (+): 6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 308, T_ALIVE = 20868
A_HP = 279604075, A_STAMINA = 191670800, A_DAMAGE = 29097666
A_HP/T_ALIVE = 13398.7, A_STAMINA/T_ALIVE = 9184.91, A_DAMAGE/T_ALIVE = 1394.37
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 10433
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 869
Wall Collisions   (-): 328
Adjancent Enemies (-): 49
Flanking Enemies  (-): 492
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3479
Wall Collisions   (+): 2627
Adjancent Enemies (+): 839
Flanking Enemies  (+): 13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 208, T_ALIVE = 12818
A_HP = 129943060, A_STAMINA = 129252960, A_DAMAGE = 17653434
A_HP/T_ALIVE = 10137.5, A_STAMINA/T_ALIVE = 10083.7, A_DAMAGE/T_ALIVE = 1377.24
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6408
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 689
Wall Collisions   (-): 245
Adjancent Enemies (-): 137
Flanking Enemies  (-): 307
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2171
Wall Collisions   (+): 1515
Adjancent Enemies (+): 641
Flanking Enemies  (+): 15
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 92, T_ALIVE = 5248
A_HP = 47003235, A_STAMINA = 69481400, A_DAMAGE = 9860620
A_HP/T_ALIVE = 8956.41, A_STAMINA/T_ALIVE = 13239.6, A_DAMAGE/T_ALIVE = 1878.93
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2623
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 295
Wall Collisions   (-): 127
Adjancent Enemies (-): 14
Flanking Enemies  (-): 154
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 804
Wall Collisions   (+): 707
Adjancent Enemies (+): 84
Flanking Enemies  (+): 13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 243, T_ALIVE = 13550
A_HP = 119954620, A_STAMINA = 145452840, A_DAMAGE = 36237840
A_HP/T_ALIVE = 8852.74, A_STAMINA/T_ALIVE = 10734.5, A_DAMAGE/T_ALIVE = 2674.38
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6774
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 600
Wall Collisions   (-): 276
Adjancent Enemies (-): 15
Flanking Enemies  (-): 309
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2201
Wall Collisions   (+): 1949
Adjancent Enemies (+): 232
Flanking Enemies  (+): 20
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 106, T_ALIVE = 7308
A_HP = 47911925, A_STAMINA = 94415880, A_DAMAGE = 12165980
A_HP/T_ALIVE = 6556.09, A_STAMINA/T_ALIVE = 12919.5, A_DAMAGE/T_ALIVE = 1664.75
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3653
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 260
Wall Collisions   (-): 107
Adjancent Enemies (-): 10
Flanking Enemies  (-): 143
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1152
Wall Collisions   (+): 1030
Adjancent Enemies (+): 111
Flanking Enemies  (+): 11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 382, T_ALIVE = 25514
A_HP = 166226545, A_STAMINA = 193820820, A_DAMAGE = 42147005
A_HP/T_ALIVE = 6515.11, A_STAMINA/T_ALIVE = 7596.65, A_DAMAGE/T_ALIVE = 1651.92
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 12756
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1132
Wall Collisions   (-): 577
Adjancent Enemies (-): 27
Flanking Enemies  (-): 528
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 4295
Wall Collisions   (+): 3866
Adjancent Enemies (+): 399
Flanking Enemies  (+): 30
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 235, T_ALIVE = 18388
A_HP = 191039980, A_STAMINA = 181071980, A_DAMAGE = 24573134
A_HP/T_ALIVE = 10389.4, A_STAMINA/T_ALIVE = 9847.29, A_DAMAGE/T_ALIVE = 1336.37
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 9193
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1604
Wall Collisions   (-): 1144
Adjancent Enemies (-): 133
Flanking Enemies  (-): 327
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3423
Wall Collisions   (+): 2937
Adjancent Enemies (+): 466
Flanking Enemies  (+): 20
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 116, T_ALIVE = 7500
A_HP = 55169790, A_STAMINA = 91232820, A_DAMAGE = 8950795
A_HP/T_ALIVE = 7355.97, A_STAMINA/T_ALIVE = 12164.4, A_DAMAGE/T_ALIVE = 1193.44
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3749
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 309
Wall Collisions   (-): 125
Adjancent Enemies (-): 14
Flanking Enemies  (-): 170
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1272
Wall Collisions   (+): 1104
Adjancent Enemies (+): 159
Flanking Enemies  (+): 9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 157, T_ALIVE = 9194
A_HP = 67965575, A_STAMINA = 104591320, A_DAMAGE = 14720105
A_HP/T_ALIVE = 7392.38, A_STAMINA/T_ALIVE = 11376, A_DAMAGE/T_ALIVE = 1601.06
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4596
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 300
Wall Collisions   (-): 101
Adjancent Enemies (-): 12
Flanking Enemies  (-): 187
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1123
Wall Collisions   (+): 894
Adjancent Enemies (+): 214
Flanking Enemies  (+): 15
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 433, T_ALIVE = 26138
A_HP = 221709760, A_STAMINA = 203544340, A_DAMAGE = 69559980
A_HP/T_ALIVE = 8482.28, A_STAMINA/T_ALIVE = 7787.3, A_DAMAGE/T_ALIVE = 2661.26
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 13068
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1891
Wall Collisions   (-): 1249
Adjancent Enemies (-): 40
Flanking Enemies  (-): 602
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 4255
Wall Collisions   (+): 3791
Adjancent Enemies (+): 430
Flanking Enemies  (+): 34
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 138, T_ALIVE = 8564
A_HP = 60765185, A_STAMINA = 107862760, A_DAMAGE = 16019695
A_HP/T_ALIVE = 7095.42, A_STAMINA/T_ALIVE = 12594.9, A_DAMAGE/T_ALIVE = 1870.59
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4281
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 296
Wall Collisions   (-): 105
Adjancent Enemies (-): 12
Flanking Enemies  (-): 179
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1307
Wall Collisions   (+): 1167
Adjancent Enemies (+): 130
Flanking Enemies  (+): 10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 326, T_ALIVE = 21558
A_HP = 238315505, A_STAMINA = 216277080, A_DAMAGE = 49215880
A_HP/T_ALIVE = 11054.6, A_STAMINA/T_ALIVE = 10032.3, A_DAMAGE/T_ALIVE = 2282.95
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 10778
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1259
Wall Collisions   (-): 789
Adjancent Enemies (-): 24
Flanking Enemies  (-): 446
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3681
Wall Collisions   (+): 3354
Adjancent Enemies (+): 307
Flanking Enemies  (+): 20
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 138, T_ALIVE = 8910
A_HP = 73101575, A_STAMINA = 95639760, A_DAMAGE = 10705123
A_HP/T_ALIVE = 8204.44, A_STAMINA/T_ALIVE = 10734, A_DAMAGE/T_ALIVE = 1201.47
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4454
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 351
Wall Collisions   (-): 82
Adjancent Enemies (-): 54
Flanking Enemies  (-): 215
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1428
Wall Collisions   (+): 953
Adjancent Enemies (+): 458
Flanking Enemies  (+): 17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 113, T_ALIVE = 7574
A_HP = 80124430, A_STAMINA = 88725820, A_DAMAGE = 11595578
A_HP/T_ALIVE = 10578.9, A_STAMINA/T_ALIVE = 11714.5, A_DAMAGE/T_ALIVE = 1530.97
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3786
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 370
Wall Collisions   (-): 88
Adjancent Enemies (-): 113
Flanking Enemies  (-): 169
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1192
Wall Collisions   (+): 789
Adjancent Enemies (+): 382
Flanking Enemies  (+): 21
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 283, T_ALIVE = 19216
A_HP = 127688115, A_STAMINA = 173799940, A_DAMAGE = 31524760
A_HP/T_ALIVE = 6644.89, A_STAMINA/T_ALIVE = 9044.54, A_DAMAGE/T_ALIVE = 1640.55
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 9607
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 861
Wall Collisions   (-): 440
Adjancent Enemies (-): 22
Flanking Enemies  (-): 399
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3139
Wall Collisions   (+): 2839
Adjancent Enemies (+): 285
Flanking Enemies  (+): 15
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 113, T_ALIVE = 8476
A_HP = 69579175, A_STAMINA = 74758080, A_DAMAGE = 4941541
A_HP/T_ALIVE = 8208.96, A_STAMINA/T_ALIVE = 8819.97, A_DAMAGE/T_ALIVE = 583.004
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4237
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 300
Wall Collisions   (-): 105
Adjancent Enemies (-): 26
Flanking Enemies  (-): 169
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1939
Wall Collisions   (+): 918
Adjancent Enemies (+): 1009
Flanking Enemies  (+): 12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 472, T_ALIVE = 36210
A_HP = 283704630, A_STAMINA = 203123320, A_DAMAGE = 87883210
A_HP/T_ALIVE = 7834.98, A_STAMINA/T_ALIVE = 5609.59, A_DAMAGE/T_ALIVE = 2427.04
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 18104
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 2345
Wall Collisions   (-): 1645
Adjancent Enemies (-): 54
Flanking Enemies  (-): 646
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 7411
Wall Collisions   (+): 6686
Adjancent Enemies (+): 688
Flanking Enemies  (+): 37
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 104, T_ALIVE = 8342
A_HP = 68679070, A_STAMINA = 88019960, A_DAMAGE = 9326262
A_HP/T_ALIVE = 8232.93, A_STAMINA/T_ALIVE = 10551.4, A_DAMAGE/T_ALIVE = 1117.99
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4170
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 291
Wall Collisions   (-): 103
Adjancent Enemies (-): 35
Flanking Enemies  (-): 153
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1663
Wall Collisions   (+): 909
Adjancent Enemies (+): 741
Flanking Enemies  (+): 13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 55, T_ALIVE = 3316
A_HP = 27255965, A_STAMINA = 46255920, A_DAMAGE = 5167372
A_HP/T_ALIVE = 8219.53, A_STAMINA/T_ALIVE = 13949.3, A_DAMAGE/T_ALIVE = 1558.31
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 1657
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 129
Wall Collisions   (-): 45
Adjancent Enemies (-): 7
Flanking Enemies  (-): 77
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 383
Wall Collisions   (+): 315
Adjancent Enemies (+): 59
Flanking Enemies  (+): 9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 82, T_ALIVE = 5182
A_HP = 47282035, A_STAMINA = 63873800, A_DAMAGE = 7048793
A_HP/T_ALIVE = 9124.28, A_STAMINA/T_ALIVE = 12326.1, A_DAMAGE/T_ALIVE = 1360.25
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2590
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 180
Wall Collisions   (-): 49
Adjancent Enemies (-): 11
Flanking Enemies  (-): 120
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 879
Wall Collisions   (+): 614
Adjancent Enemies (+): 252
Flanking Enemies  (+): 13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 232, T_ALIVE = 14946
A_HP = 105476055, A_STAMINA = 162108220, A_DAMAGE = 27935995
A_HP/T_ALIVE = 7057.14, A_STAMINA/T_ALIVE = 10846.3, A_DAMAGE/T_ALIVE = 1869.13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7472
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 665
Wall Collisions   (-): 326
Adjancent Enemies (-): 20
Flanking Enemies  (-): 319
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2436
Wall Collisions   (+): 2191
Adjancent Enemies (+): 230
Flanking Enemies  (+): 15
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 138, T_ALIVE = 8280
A_HP = 68879870, A_STAMINA = 93114840, A_DAMAGE = 11511952
A_HP/T_ALIVE = 8318.82, A_STAMINA/T_ALIVE = 11245.8, A_DAMAGE/T_ALIVE = 1390.33
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4139
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 408
Wall Collisions   (-): 130
Adjancent Enemies (-): 65
Flanking Enemies  (-): 213
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1378
Wall Collisions   (+): 1054
Adjancent Enemies (+): 305
Flanking Enemies  (+): 19
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 71, T_ALIVE = 5284
A_HP = 50320615, A_STAMINA = 65057400, A_DAMAGE = 6991200
A_HP/T_ALIVE = 9523.2, A_STAMINA/T_ALIVE = 12312.1, A_DAMAGE/T_ALIVE = 1323.09
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2641
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 242
Wall Collisions   (-): 76
Adjancent Enemies (-): 54
Flanking Enemies  (-): 112
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 778
Wall Collisions   (+): 436
Adjancent Enemies (+): 329
Flanking Enemies  (+): 13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 269, T_ALIVE = 21334
A_HP = 165402505, A_STAMINA = 189545320, A_DAMAGE = 30525877
A_HP/T_ALIVE = 7753, A_STAMINA/T_ALIVE = 8884.66, A_DAMAGE/T_ALIVE = 1430.86
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 10666
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1026
Wall Collisions   (-): 528
Adjancent Enemies (-): 106
Flanking Enemies  (-): 392
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 4321
Wall Collisions   (+): 3097
Adjancent Enemies (+): 1202
Flanking Enemies  (+): 22
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 169, T_ALIVE = 10992
A_HP = 89695750, A_STAMINA = 123298220, A_DAMAGE = 14312532
A_HP/T_ALIVE = 8160.09, A_STAMINA/T_ALIVE = 11217.1, A_DAMAGE/T_ALIVE = 1302.09
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5495
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 532
Wall Collisions   (-): 269
Adjancent Enemies (-): 41
Flanking Enemies  (-): 222
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1988
Wall Collisions   (+): 1619
Adjancent Enemies (+): 342
Flanking Enemies  (+): 27
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 450, T_ALIVE = 27738
A_HP = 355187025, A_STAMINA = 208695000, A_DAMAGE = 66501939
A_HP/T_ALIVE = 12805.1, A_STAMINA/T_ALIVE = 7523.79, A_DAMAGE/T_ALIVE = 2397.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 13868
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1609
Wall Collisions   (-): 921
Adjancent Enemies (-): 65
Flanking Enemies  (-): 623
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 4934
Wall Collisions   (+): 4341
Adjancent Enemies (+): 543
Flanking Enemies  (+): 50
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 135, T_ALIVE = 7182
A_HP = 67982255, A_STAMINA = 87923460, A_DAMAGE = 14857275
A_HP/T_ALIVE = 9465.64, A_STAMINA/T_ALIVE = 12242.2, A_DAMAGE/T_ALIVE = 2068.68
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3590
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 279
Wall Collisions   (-): 60
Adjancent Enemies (-): 18
Flanking Enemies  (-): 201
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 800
Wall Collisions   (+): 646
Adjancent Enemies (+): 143
Flanking Enemies  (+): 11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 250, T_ALIVE = 16208
A_HP = 149203775, A_STAMINA = 178183380, A_DAMAGE = 38806140
A_HP/T_ALIVE = 9205.56, A_STAMINA/T_ALIVE = 10993.5, A_DAMAGE/T_ALIVE = 2394.26
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 8103
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 684
Wall Collisions   (-): 287
Adjancent Enemies (-): 20
Flanking Enemies  (-): 377
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2525
Wall Collisions   (+): 2274
Adjancent Enemies (+): 236
Flanking Enemies  (+): 15
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 290, T_ALIVE = 19216
A_HP = 142114680, A_STAMINA = 177248880, A_DAMAGE = 30057648
A_HP/T_ALIVE = 7395.64, A_STAMINA/T_ALIVE = 9224.03, A_DAMAGE/T_ALIVE = 1564.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 9607
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 956
Wall Collisions   (-): 523
Adjancent Enemies (-): 20
Flanking Enemies  (-): 413
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3532
Wall Collisions   (+): 3121
Adjancent Enemies (+): 380
Flanking Enemies  (+): 31
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 128, T_ALIVE = 6772
A_HP = 68375250, A_STAMINA = 86742580, A_DAMAGE = 17419155
A_HP/T_ALIVE = 10096.8, A_STAMINA/T_ALIVE = 12809, A_DAMAGE/T_ALIVE = 2572.23
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3385
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 275
Wall Collisions   (-): 70
Adjancent Enemies (-): 13
Flanking Enemies  (-): 192
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 848
Wall Collisions   (+): 717
Adjancent Enemies (+): 118
Flanking Enemies  (+): 13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 135, T_ALIVE = 7464
A_HP = 75690495, A_STAMINA = 87681960, A_DAMAGE = 12801434
A_HP/T_ALIVE = 10140.7, A_STAMINA/T_ALIVE = 11747.3, A_DAMAGE/T_ALIVE = 1715.09
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3731
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 351
Wall Collisions   (-): 81
Adjancent Enemies (-): 72
Flanking Enemies  (-): 198
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1090
Wall Collisions   (+): 791
Adjancent Enemies (+): 285
Flanking Enemies  (+): 14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 129, T_ALIVE = 8476
A_HP = 59582490, A_STAMINA = 98293000, A_DAMAGE = 10195050
A_HP/T_ALIVE = 7029.55, A_STAMINA/T_ALIVE = 11596.6, A_DAMAGE/T_ALIVE = 1202.81
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4237
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 360
Wall Collisions   (-): 174
Adjancent Enemies (-): 17
Flanking Enemies  (-): 169
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1307
Wall Collisions   (+): 1107
Adjancent Enemies (+): 190
Flanking Enemies  (+): 10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 126, T_ALIVE = 7692
A_HP = 70246020, A_STAMINA = 99153200, A_DAMAGE = 16120560
A_HP/T_ALIVE = 9132.35, A_STAMINA/T_ALIVE = 12890.4, A_DAMAGE/T_ALIVE = 2095.76
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3845
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 298
Wall Collisions   (-): 116
Adjancent Enemies (-): 10
Flanking Enemies  (-): 172
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1080
Wall Collisions   (+): 946
Adjancent Enemies (+): 120
Flanking Enemies  (+): 14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 96, T_ALIVE = 5014
A_HP = 39557760, A_STAMINA = 66501920, A_DAMAGE = 10243135
A_HP/T_ALIVE = 7889.46, A_STAMINA/T_ALIVE = 13263.2, A_DAMAGE/T_ALIVE = 2042.91
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2506
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 221
Wall Collisions   (-): 70
Adjancent Enemies (-): 10
Flanking Enemies  (-): 141
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 667
Wall Collisions   (+): 570
Adjancent Enemies (+): 88
Flanking Enemies  (+): 9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 242, T_ALIVE = 13166
A_HP = 96783750, A_STAMINA = 140121580, A_DAMAGE = 36873585
A_HP/T_ALIVE = 7351.04, A_STAMINA/T_ALIVE = 10642.7, A_DAMAGE/T_ALIVE = 2800.67
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6582
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 534
Wall Collisions   (-): 199
Adjancent Enemies (-): 17
Flanking Enemies  (-): 318
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1880
Wall Collisions   (+): 1633
Adjancent Enemies (+): 229
Flanking Enemies  (+): 18
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 106, T_ALIVE = 7232
A_HP = 66048805, A_STAMINA = 89119180, A_DAMAGE = 10728323
A_HP/T_ALIVE = 9132.85, A_STAMINA/T_ALIVE = 12322.9, A_DAMAGE/T_ALIVE = 1483.45
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3615
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 436
Wall Collisions   (-): 170
Adjancent Enemies (-): 121
Flanking Enemies  (-): 145
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1152
Wall Collisions   (+): 871
Adjancent Enemies (+): 270
Flanking Enemies  (+): 11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 126, T_ALIVE = 6722
A_HP = 52043435, A_STAMINA = 84516800, A_DAMAGE = 13870835
A_HP/T_ALIVE = 7742.25, A_STAMINA/T_ALIVE = 12573.2, A_DAMAGE/T_ALIVE = 2063.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3360
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 280
Wall Collisions   (-): 97
Adjancent Enemies (-): 7
Flanking Enemies  (-): 176
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 963
Wall Collisions   (+): 818
Adjancent Enemies (+): 129
Flanking Enemies  (+): 16
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 134, T_ALIVE = 8380
A_HP = 75770960, A_STAMINA = 96652760, A_DAMAGE = 10551861
A_HP/T_ALIVE = 9041.88, A_STAMINA/T_ALIVE = 11533.7, A_DAMAGE/T_ALIVE = 1259.17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4189
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 433
Wall Collisions   (-): 231
Adjancent Enemies (-): 22
Flanking Enemies  (-): 180
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1264
Wall Collisions   (+): 1011
Adjancent Enemies (+): 247
Flanking Enemies  (+): 6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 253, T_ALIVE = 17084
A_HP = 202331585, A_STAMINA = 183499180, A_DAMAGE = 42169843
A_HP/T_ALIVE = 11843.3, A_STAMINA/T_ALIVE = 10741, A_DAMAGE/T_ALIVE = 2468.38
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 8541
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 737
Wall Collisions   (-): 379
Adjancent Enemies (-): 18
Flanking Enemies  (-): 340
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2857
Wall Collisions   (+): 2408
Adjancent Enemies (+): 428
Flanking Enemies  (+): 21
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 110, T_ALIVE = 7510
A_HP = 67591090, A_STAMINA = 82850060, A_DAMAGE = 8862514
A_HP/T_ALIVE = 9000.15, A_STAMINA/T_ALIVE = 11032, A_DAMAGE/T_ALIVE = 1180.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3754
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 300
Wall Collisions   (-): 108
Adjancent Enemies (-): 24
Flanking Enemies  (-): 168
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1335
Wall Collisions   (+): 915
Adjancent Enemies (+): 402
Flanking Enemies  (+): 18
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 152, T_ALIVE = 8364
A_HP = 64193095, A_STAMINA = 91137640, A_DAMAGE = 10832232
A_HP/T_ALIVE = 7674.93, A_STAMINA/T_ALIVE = 10896.4, A_DAMAGE/T_ALIVE = 1295.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4181
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 423
Wall Collisions   (-): 187
Adjancent Enemies (-): 17
Flanking Enemies  (-): 219
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1266
Wall Collisions   (+): 985
Adjancent Enemies (+): 271
Flanking Enemies  (+): 10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 138, T_ALIVE = 12596
A_HP = 164474195, A_STAMINA = 117066100, A_DAMAGE = 18316878
A_HP/T_ALIVE = 13057.7, A_STAMINA/T_ALIVE = 9293.91, A_DAMAGE/T_ALIVE = 1454.18
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6297
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1061
Wall Collisions   (-): 257
Adjancent Enemies (-): 586
Flanking Enemies  (-): 218
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2143
Wall Collisions   (+): 1270
Adjancent Enemies (+): 840
Flanking Enemies  (+): 33
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 158, T_ALIVE = 9210
A_HP = 70766335, A_STAMINA = 113333000, A_DAMAGE = 18866000
A_HP/T_ALIVE = 7683.64, A_STAMINA/T_ALIVE = 12305.4, A_DAMAGE/T_ALIVE = 2048.43
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4604
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 430
Wall Collisions   (-): 207
Adjancent Enemies (-): 12
Flanking Enemies  (-): 211
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1433
Wall Collisions   (+): 1266
Adjancent Enemies (+): 155
Flanking Enemies  (+): 12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 236, T_ALIVE = 14936
A_HP = 115347670, A_STAMINA = 162358580, A_DAMAGE = 36563155
A_HP/T_ALIVE = 7722.8, A_STAMINA/T_ALIVE = 10870.3, A_DAMAGE/T_ALIVE = 2447.99
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7467
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 848
Wall Collisions   (-): 481
Adjancent Enemies (-): 12
Flanking Enemies  (-): 355
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2189
Wall Collisions   (+): 1939
Adjancent Enemies (+): 228
Flanking Enemies  (+): 22
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 283, T_ALIVE = 19630
A_HP = 123788675, A_STAMINA = 188863260, A_DAMAGE = 40287900
A_HP/T_ALIVE = 6306.1, A_STAMINA/T_ALIVE = 9621.15, A_DAMAGE/T_ALIVE = 2052.36
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 9814
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 892
Wall Collisions   (-): 490
Adjancent Enemies (-): 19
Flanking Enemies  (-): 383
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3518
Wall Collisions   (+): 3202
Adjancent Enemies (+): 299
Flanking Enemies  (+): 17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 166, T_ALIVE = 11382
A_HP = 74928435, A_STAMINA = 125171080, A_DAMAGE = 17136301
A_HP/T_ALIVE = 6583.06, A_STAMINA/T_ALIVE = 10997.3, A_DAMAGE/T_ALIVE = 1505.56
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5690
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 453
Wall Collisions   (-): 130
Adjancent Enemies (-): 47
Flanking Enemies  (-): 276
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1870
Wall Collisions   (+): 1318
Adjancent Enemies (+): 542
Flanking Enemies  (+): 10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/pre-ewc+kd-backup/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 128, T_ALIVE = 11134
A_HP = 47186085, A_STAMINA = 116958140, A_DAMAGE = 9824990
A_HP/T_ALIVE = 4238.02, A_STAMINA/T_ALIVE = 10504.6, A_DAMAGE/T_ALIVE = 882.431
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5566
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 816
Wall Collisions   (-): 603
Adjancent Enemies (-): 7
Flanking Enemies  (-): 206
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1972
Wall Collisions   (+): 1758
Adjancent Enemies (+): 211
Flanking Enemies  (+): 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

------------------

## backup [MASK1]:

```
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 162, T_ALIVE = 8788
A_HP = 55440280, A_STAMINA = 101083620, A_DAMAGE = 24007895
A_HP/T_ALIVE = 6308.63, A_STAMINA/T_ALIVE = 11502.5, A_DAMAGE/T_ALIVE = 2731.9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4393
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 627
Wall Collisions   (-): 416
Adjancent Enemies (-): 24
Flanking Enemies  (-): 187
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1418
Wall Collisions   (+): 1237
Adjancent Enemies (+): 142
Flanking Enemies  (+): 39
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 226, T_ALIVE = 12744
A_HP = 69347230, A_STAMINA = 128874840, A_DAMAGE = 17438375
A_HP/T_ALIVE = 5441.56, A_STAMINA/T_ALIVE = 10112.6, A_DAMAGE/T_ALIVE = 1368.36
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6371
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 591
Wall Collisions   (-): 285
Adjancent Enemies (-): 43
Flanking Enemies  (-): 263
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1763
Wall Collisions   (+): 1471
Adjancent Enemies (+): 225
Flanking Enemies  (+): 67
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 112, T_ALIVE = 5934
A_HP = 41757495, A_STAMINA = 75486460, A_DAMAGE = 11019665
A_HP/T_ALIVE = 7036.99, A_STAMINA/T_ALIVE = 12721, A_DAMAGE/T_ALIVE = 1857.04
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2966
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 187
Wall Collisions   (-): 34
Adjancent Enemies (-): 18
Flanking Enemies  (-): 135
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 699
Wall Collisions   (+): 566
Adjancent Enemies (+): 101
Flanking Enemies  (+): 32
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 164, T_ALIVE = 8404
A_HP = 60909340, A_STAMINA = 95218720, A_DAMAGE = 12498625
A_HP/T_ALIVE = 7247.66, A_STAMINA/T_ALIVE = 11330.2, A_DAMAGE/T_ALIVE = 1487.22
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4201
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 428
Wall Collisions   (-): 156
Adjancent Enemies (-): 32
Flanking Enemies  (-): 240
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 949
Wall Collisions   (+): 745
Adjancent Enemies (+): 167
Flanking Enemies  (+): 37
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 417, T_ALIVE = 23552
A_HP = 274677990, A_STAMINA = 175911840, A_DAMAGE = 73229445
A_HP/T_ALIVE = 11662.6, A_STAMINA/T_ALIVE = 7469.08, A_DAMAGE/T_ALIVE = 3109.27
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 11775
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1288
Wall Collisions   (-): 670
Adjancent Enemies (-): 60
Flanking Enemies  (-): 558
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3485
Wall Collisions   (+): 3021
Adjancent Enemies (+): 385
Flanking Enemies  (+): 79
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 106, T_ALIVE = 7564
A_HP = 68418415, A_STAMINA = 86251100, A_DAMAGE = 11773959
A_HP/T_ALIVE = 9045.27, A_STAMINA/T_ALIVE = 11402.8, A_DAMAGE/T_ALIVE = 1556.58
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3781
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 830
Wall Collisions   (-): 499
Adjancent Enemies (-): 177
Flanking Enemies  (-): 154
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1291
Wall Collisions   (+): 1029
Adjancent Enemies (+): 218
Flanking Enemies  (+): 44
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 115, T_ALIVE = 5786
A_HP = 48616210, A_STAMINA = 73525120, A_DAMAGE = 16457200
A_HP/T_ALIVE = 8402.39, A_STAMINA/T_ALIVE = 12707.4, A_DAMAGE/T_ALIVE = 2844.31
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2892
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 372
Wall Collisions   (-): 185
Adjancent Enemies (-): 7
Flanking Enemies  (-): 180
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 807
Wall Collisions   (+): 694
Adjancent Enemies (+): 108
Flanking Enemies  (+): 5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 215, T_ALIVE = 10566
A_HP = 77049240, A_STAMINA = 117707580, A_DAMAGE = 24833670
A_HP/T_ALIVE = 7292.19, A_STAMINA/T_ALIVE = 11140.2, A_DAMAGE/T_ALIVE = 2350.34
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5282
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 809
Wall Collisions   (-): 459
Adjancent Enemies (-): 52
Flanking Enemies  (-): 298
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1381
Wall Collisions   (+): 1154
Adjancent Enemies (+): 178
Flanking Enemies  (+): 49
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 237, T_ALIVE = 13318
A_HP = 77571835, A_STAMINA = 144489840, A_DAMAGE = 23721620
A_HP/T_ALIVE = 5824.59, A_STAMINA/T_ALIVE = 10849.2, A_DAMAGE/T_ALIVE = 1781.17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6658
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 696
Wall Collisions   (-): 331
Adjancent Enemies (-): 40
Flanking Enemies  (-): 325
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2121
Wall Collisions   (+): 1846
Adjancent Enemies (+): 212
Flanking Enemies  (+): 63
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 213, T_ALIVE = 10780
A_HP = 70782270, A_STAMINA = 116846580, A_DAMAGE = 31270750
A_HP/T_ALIVE = 6566.07, A_STAMINA/T_ALIVE = 10839.2, A_DAMAGE/T_ALIVE = 2900.81
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5389
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 474
Wall Collisions   (-): 159
Adjancent Enemies (-): 31
Flanking Enemies  (-): 284
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1142
Wall Collisions   (+): 904
Adjancent Enemies (+): 188
Flanking Enemies  (+): 50
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 205, T_ALIVE = 11512
A_HP = 83741990, A_STAMINA = 131614380, A_DAMAGE = 24409555
A_HP/T_ALIVE = 7274.32, A_STAMINA/T_ALIVE = 11432.8, A_DAMAGE/T_ALIVE = 2120.36
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5755
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 850
Wall Collisions   (-): 536
Adjancent Enemies (-): 36
Flanking Enemies  (-): 278
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1585
Wall Collisions   (+): 1357
Adjancent Enemies (+): 176
Flanking Enemies  (+): 52
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 181, T_ALIVE = 8514
A_HP = 65330500, A_STAMINA = 100067220, A_DAMAGE = 19789900
A_HP/T_ALIVE = 7673.3, A_STAMINA/T_ALIVE = 11753.3, A_DAMAGE/T_ALIVE = 2324.4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4256
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 444
Wall Collisions   (-): 183
Adjancent Enemies (-): 34
Flanking Enemies  (-): 227
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1083
Wall Collisions   (+): 879
Adjancent Enemies (+): 153
Flanking Enemies  (+): 51
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 74, T_ALIVE = 3400
A_HP = 29121370, A_STAMINA = 46367180, A_DAMAGE = 8694515
A_HP/T_ALIVE = 8565.11, A_STAMINA/T_ALIVE = 13637.4, A_DAMAGE/T_ALIVE = 2557.21
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 1699
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 145
Wall Collisions   (-): 22
Adjancent Enemies (-): 14
Flanking Enemies  (-): 109
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 440
Wall Collisions   (+): 355
Adjancent Enemies (+): 61
Flanking Enemies  (+): 24
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 182, T_ALIVE = 12924
A_HP = 70357445, A_STAMINA = 142340680, A_DAMAGE = 25072720
A_HP/T_ALIVE = 5443.94, A_STAMINA/T_ALIVE = 11013.7, A_DAMAGE/T_ALIVE = 1940.01
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6461
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1180
Wall Collisions   (-): 903
Adjancent Enemies (-): 36
Flanking Enemies  (-): 241
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1808
Wall Collisions   (+): 1603
Adjancent Enemies (+): 164
Flanking Enemies  (+): 41
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 138, T_ALIVE = 7716
A_HP = 60779930, A_STAMINA = 95451700, A_DAMAGE = 16549075
A_HP/T_ALIVE = 7877.13, A_STAMINA/T_ALIVE = 12370.6, A_DAMAGE/T_ALIVE = 2144.77
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3857
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 414
Wall Collisions   (-): 185
Adjancent Enemies (-): 28
Flanking Enemies  (-): 201
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 975
Wall Collisions   (+): 815
Adjancent Enemies (+): 112
Flanking Enemies  (+): 48
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 144, T_ALIVE = 9254
A_HP = 73881815, A_STAMINA = 111588800, A_DAMAGE = 13777307
A_HP/T_ALIVE = 7983.77, A_STAMINA/T_ALIVE = 12058.4, A_DAMAGE/T_ALIVE = 1488.79
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4626
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 373
Wall Collisions   (-): 138
Adjancent Enemies (-): 38
Flanking Enemies  (-): 197
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1574
Wall Collisions   (+): 1348
Adjancent Enemies (+): 196
Flanking Enemies  (+): 30
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 230, T_ALIVE = 13118
A_HP = 106746100, A_STAMINA = 139818360, A_DAMAGE = 29838840
A_HP/T_ALIVE = 8137.38, A_STAMINA/T_ALIVE = 10658.5, A_DAMAGE/T_ALIVE = 2274.65
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6558
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 705
Wall Collisions   (-): 373
Adjancent Enemies (-): 36
Flanking Enemies  (-): 296
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1796
Wall Collisions   (+): 1541
Adjancent Enemies (+): 202
Flanking Enemies  (+): 53
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 216, T_ALIVE = 14642
A_HP = 101172675, A_STAMINA = 155040920, A_DAMAGE = 29986842
A_HP/T_ALIVE = 6909.76, A_STAMINA/T_ALIVE = 10588.8, A_DAMAGE/T_ALIVE = 2048
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7320
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1253
Wall Collisions   (-): 782
Adjancent Enemies (-): 180
Flanking Enemies  (-): 291
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2379
Wall Collisions   (+): 1983
Adjancent Enemies (+): 328
Flanking Enemies  (+): 68
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 91, T_ALIVE = 6682
A_HP = 64768960, A_STAMINA = 78065500, A_DAMAGE = 10013260
A_HP/T_ALIVE = 9693.05, A_STAMINA/T_ALIVE = 11683, A_DAMAGE/T_ALIVE = 1498.54
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3340
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 410
Wall Collisions   (-): 91
Adjancent Enemies (-): 157
Flanking Enemies  (-): 162
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 876
Wall Collisions   (+): 547
Adjancent Enemies (+): 303
Flanking Enemies  (+): 26
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 147, T_ALIVE = 8414
A_HP = 90813550, A_STAMINA = 101930420, A_DAMAGE = 17165295
A_HP/T_ALIVE = 10793.1, A_STAMINA/T_ALIVE = 12114.4, A_DAMAGE/T_ALIVE = 2040.09
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4206
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 406
Wall Collisions   (-): 207
Adjancent Enemies (-): 39
Flanking Enemies  (-): 160
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1318
Wall Collisions   (+): 1151
Adjancent Enemies (+): 122
Flanking Enemies  (+): 45
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 116, T_ALIVE = 6632
A_HP = 53288495, A_STAMINA = 84606900, A_DAMAGE = 15671535
A_HP/T_ALIVE = 8035.06, A_STAMINA/T_ALIVE = 12757.4, A_DAMAGE/T_ALIVE = 2363.02
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3315
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 263
Wall Collisions   (-): 108
Adjancent Enemies (-): 24
Flanking Enemies  (-): 131
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 667
Wall Collisions   (+): 546
Adjancent Enemies (+): 94
Flanking Enemies  (+): 27
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 152, T_ALIVE = 7652
A_HP = 56202065, A_STAMINA = 93237780, A_DAMAGE = 18794740
A_HP/T_ALIVE = 7344.75, A_STAMINA/T_ALIVE = 12184.8, A_DAMAGE/T_ALIVE = 2456.19
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3825
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 308
Wall Collisions   (-): 77
Adjancent Enemies (-): 34
Flanking Enemies  (-): 197
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 894
Wall Collisions   (+): 734
Adjancent Enemies (+): 120
Flanking Enemies  (+): 40
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 59, T_ALIVE = 4504
A_HP = 41509495, A_STAMINA = 39256480, A_DAMAGE = 5246998
A_HP/T_ALIVE = 9216.14, A_STAMINA/T_ALIVE = 8715.91, A_DAMAGE/T_ALIVE = 1164.96
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2251
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 231
Wall Collisions   (-): 31
Adjancent Enemies (-): 105
Flanking Enemies  (-): 95
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 993
Wall Collisions   (+): 224
Adjancent Enemies (+): 758
Flanking Enemies  (+): 11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 267, T_ALIVE = 12782
A_HP = 102789100, A_STAMINA = 128184200, A_DAMAGE = 32704415
A_HP/T_ALIVE = 8041.71, A_STAMINA/T_ALIVE = 10028.5, A_DAMAGE/T_ALIVE = 2558.63
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6390
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 562
Wall Collisions   (-): 176
Adjancent Enemies (-): 38
Flanking Enemies  (-): 348
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1609
Wall Collisions   (+): 1334
Adjancent Enemies (+): 237
Flanking Enemies  (+): 38
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 256, T_ALIVE = 15244
A_HP = 107310060, A_STAMINA = 161976560, A_DAMAGE = 42619045
A_HP/T_ALIVE = 7039.49, A_STAMINA/T_ALIVE = 10625.6, A_DAMAGE/T_ALIVE = 2795.79
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7621
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1056
Wall Collisions   (-): 709
Adjancent Enemies (-): 42
Flanking Enemies  (-): 305
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2213
Wall Collisions   (+): 1958
Adjancent Enemies (+): 218
Flanking Enemies  (+): 37
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 76, T_ALIVE = 3774
A_HP = 32025610, A_STAMINA = 50670500, A_DAMAGE = 8397605
A_HP/T_ALIVE = 8485.85, A_STAMINA/T_ALIVE = 13426.2, A_DAMAGE/T_ALIVE = 2225.12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 1886
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 196
Wall Collisions   (-): 61
Adjancent Enemies (-): 14
Flanking Enemies  (-): 121
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 484
Wall Collisions   (+): 390
Adjancent Enemies (+): 68
Flanking Enemies  (+): 26
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 126, T_ALIVE = 6124
A_HP = 50815625, A_STAMINA = 78290940, A_DAMAGE = 12127060
A_HP/T_ALIVE = 8297.78, A_STAMINA/T_ALIVE = 12784.3, A_DAMAGE/T_ALIVE = 1980.25
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3061
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 255
Wall Collisions   (-): 60
Adjancent Enemies (-): 27
Flanking Enemies  (-): 168
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 766
Wall Collisions   (+): 623
Adjancent Enemies (+): 107
Flanking Enemies  (+): 36
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 293, T_ALIVE = 14480
A_HP = 120172200, A_STAMINA = 134062680, A_DAMAGE = 28456165
A_HP/T_ALIVE = 8299.19, A_STAMINA/T_ALIVE = 9258.47, A_DAMAGE/T_ALIVE = 1965.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7239
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 710
Wall Collisions   (-): 270
Adjancent Enemies (-): 56
Flanking Enemies  (-): 384
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2108
Wall Collisions   (+): 1757
Adjancent Enemies (+): 284
Flanking Enemies  (+): 67
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 211, T_ALIVE = 11282
A_HP = 103865065, A_STAMINA = 119609160, A_DAMAGE = 20787425
A_HP/T_ALIVE = 9206.26, A_STAMINA/T_ALIVE = 10601.8, A_DAMAGE/T_ALIVE = 1842.53
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5640
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 700
Wall Collisions   (-): 360
Adjancent Enemies (-): 64
Flanking Enemies  (-): 276
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1499
Wall Collisions   (+): 1246
Adjancent Enemies (+): 198
Flanking Enemies  (+): 55
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 116, T_ALIVE = 6016
A_HP = 57908205, A_STAMINA = 77488920, A_DAMAGE = 12031875
A_HP/T_ALIVE = 9625.7, A_STAMINA/T_ALIVE = 12880.5, A_DAMAGE/T_ALIVE = 1999.98
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3007
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 393
Wall Collisions   (-): 237
Adjancent Enemies (-): 18
Flanking Enemies  (-): 138
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 804
Wall Collisions   (+): 651
Adjancent Enemies (+): 120
Flanking Enemies  (+): 33
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 89, T_ALIVE = 4240
A_HP = 37143325, A_STAMINA = 55295360, A_DAMAGE = 6760645
A_HP/T_ALIVE = 8760.22, A_STAMINA/T_ALIVE = 13041.4, A_DAMAGE/T_ALIVE = 1594.49
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2119
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 190
Wall Collisions   (-): 42
Adjancent Enemies (-): 31
Flanking Enemies  (-): 117
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 441
Wall Collisions   (+): 323
Adjancent Enemies (+): 83
Flanking Enemies  (+): 35
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 60, T_ALIVE = 3524
A_HP = 28563845, A_STAMINA = 47636720, A_DAMAGE = 4652721
A_HP/T_ALIVE = 8105.52, A_STAMINA/T_ALIVE = 13517.8, A_DAMAGE/T_ALIVE = 1320.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 1761
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 151
Wall Collisions   (-): 27
Adjancent Enemies (-): 46
Flanking Enemies  (-): 78
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 369
Wall Collisions   (+): 268
Adjancent Enemies (+): 77
Flanking Enemies  (+): 24
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 192, T_ALIVE = 11528
A_HP = 120783270, A_STAMINA = 130710180, A_DAMAGE = 20039253
A_HP/T_ALIVE = 10477.4, A_STAMINA/T_ALIVE = 11338.5, A_DAMAGE/T_ALIVE = 1738.31
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5763
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 632
Wall Collisions   (-): 308
Adjancent Enemies (-): 88
Flanking Enemies  (-): 236
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1805
Wall Collisions   (+): 1430
Adjancent Enemies (+): 339
Flanking Enemies  (+): 36
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 123, T_ALIVE = 8606
A_HP = 75664555, A_STAMINA = 96308160, A_DAMAGE = 13646747
A_HP/T_ALIVE = 8792.07, A_STAMINA/T_ALIVE = 11190.8, A_DAMAGE/T_ALIVE = 1585.72
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4302
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 440
Wall Collisions   (-): 93
Adjancent Enemies (-): 167
Flanking Enemies  (-): 180
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1366
Wall Collisions   (+): 977
Adjancent Enemies (+): 359
Flanking Enemies  (+): 30
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 94, T_ALIVE = 4536
A_HP = 40220000, A_STAMINA = 60744440, A_DAMAGE = 9801670
A_HP/T_ALIVE = 8866.84, A_STAMINA/T_ALIVE = 13391.6, A_DAMAGE/T_ALIVE = 2160.86
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2267
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 174
Wall Collisions   (-): 30
Adjancent Enemies (-): 21
Flanking Enemies  (-): 123
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 428
Wall Collisions   (+): 325
Adjancent Enemies (+): 79
Flanking Enemies  (+): 24
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 166, T_ALIVE = 8296
A_HP = 75343355, A_STAMINA = 100768020, A_DAMAGE = 21780410
A_HP/T_ALIVE = 9081.89, A_STAMINA/T_ALIVE = 12146.6, A_DAMAGE/T_ALIVE = 2625.41
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4147
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 389
Wall Collisions   (-): 102
Adjancent Enemies (-): 30
Flanking Enemies  (-): 257
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1102
Wall Collisions   (+): 916
Adjancent Enemies (+): 137
Flanking Enemies  (+): 49
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 95, T_ALIVE = 6628
A_HP = 44248525, A_STAMINA = 86285120, A_DAMAGE = 12404185
A_HP/T_ALIVE = 6676, A_STAMINA/T_ALIVE = 13018.3, A_DAMAGE/T_ALIVE = 1871.48
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3313
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 522
Wall Collisions   (-): 371
Adjancent Enemies (-): 19
Flanking Enemies  (-): 132
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1029
Wall Collisions   (+): 920
Adjancent Enemies (+): 84
Flanking Enemies  (+): 25
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 384, T_ALIVE = 21978
A_HP = 171618690, A_STAMINA = 175905840, A_DAMAGE = 53613645
A_HP/T_ALIVE = 7808.66, A_STAMINA/T_ALIVE = 8003.72, A_DAMAGE/T_ALIVE = 2439.42
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 10988
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1418
Wall Collisions   (-): 778
Adjancent Enemies (-): 62
Flanking Enemies  (-): 578
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3553
Wall Collisions   (+): 3139
Adjancent Enemies (+): 331
Flanking Enemies  (+): 83
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 82, T_ALIVE = 5642
A_HP = 43785600, A_STAMINA = 64397540, A_DAMAGE = 7568659
A_HP/T_ALIVE = 7760.65, A_STAMINA/T_ALIVE = 11414, A_DAMAGE/T_ALIVE = 1341.49
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2820
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 285
Wall Collisions   (-): 37
Adjancent Enemies (-): 140
Flanking Enemies  (-): 108
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1041
Wall Collisions   (+): 592
Adjancent Enemies (+): 420
Flanking Enemies  (+): 29
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 151, T_ALIVE = 10116
A_HP = 84946115, A_STAMINA = 121148460, A_DAMAGE = 18075205
A_HP/T_ALIVE = 8397.2, A_STAMINA/T_ALIVE = 11975.9, A_DAMAGE/T_ALIVE = 1786.79
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5057
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 899
Wall Collisions   (-): 687
Adjancent Enemies (-): 32
Flanking Enemies  (-): 180
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1583
Wall Collisions   (+): 1414
Adjancent Enemies (+): 133
Flanking Enemies  (+): 36
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 192, T_ALIVE = 10678
A_HP = 100266780, A_STAMINA = 127527100, A_DAMAGE = 27250680
A_HP/T_ALIVE = 9390.03, A_STAMINA/T_ALIVE = 11943, A_DAMAGE/T_ALIVE = 2552.04
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5338
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 483
Wall Collisions   (-): 200
Adjancent Enemies (-): 33
Flanking Enemies  (-): 250
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1736
Wall Collisions   (+): 1519
Adjancent Enemies (+): 161
Flanking Enemies  (+): 56
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 118, T_ALIVE = 6022
A_HP = 41021100, A_STAMINA = 74831420, A_DAMAGE = 10413895
A_HP/T_ALIVE = 6811.87, A_STAMINA/T_ALIVE = 12426.3, A_DAMAGE/T_ALIVE = 1729.31
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3010
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 240
Wall Collisions   (-): 69
Adjancent Enemies (-): 28
Flanking Enemies  (-): 143
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 805
Wall Collisions   (+): 665
Adjancent Enemies (+): 109
Flanking Enemies  (+): 31
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 117, T_ALIVE = 7018
A_HP = 50592070, A_STAMINA = 89370860, A_DAMAGE = 15375075
A_HP/T_ALIVE = 7208.9, A_STAMINA/T_ALIVE = 12734.5, A_DAMAGE/T_ALIVE = 2190.81
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3508
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 244
Wall Collisions   (-): 72
Adjancent Enemies (-): 15
Flanking Enemies  (-): 157
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 779
Wall Collisions   (+): 649
Adjancent Enemies (+): 108
Flanking Enemies  (+): 22
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 113, T_ALIVE = 6788
A_HP = 62800985, A_STAMINA = 78692160, A_DAMAGE = 9971602
A_HP/T_ALIVE = 9251.77, A_STAMINA/T_ALIVE = 11592.8, A_DAMAGE/T_ALIVE = 1469
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3393
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 467
Wall Collisions   (-): 218
Adjancent Enemies (-): 106
Flanking Enemies  (-): 143
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 990
Wall Collisions   (+): 662
Adjancent Enemies (+): 296
Flanking Enemies  (+): 32
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 247, T_ALIVE = 14416
A_HP = 91893005, A_STAMINA = 136341920, A_DAMAGE = 22466759
A_HP/T_ALIVE = 6374.38, A_STAMINA/T_ALIVE = 9457.68, A_DAMAGE/T_ALIVE = 1558.46
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7207
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1048
Wall Collisions   (-): 446
Adjancent Enemies (-): 271
Flanking Enemies  (-): 331
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2173
Wall Collisions   (+): 1752
Adjancent Enemies (+): 345
Flanking Enemies  (+): 76
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 165, T_ALIVE = 8266
A_HP = 62963530, A_STAMINA = 97977240, A_DAMAGE = 20907955
A_HP/T_ALIVE = 7617.17, A_STAMINA/T_ALIVE = 11853, A_DAMAGE/T_ALIVE = 2529.39
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4132
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 393
Wall Collisions   (-): 146
Adjancent Enemies (-): 40
Flanking Enemies  (-): 207
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1009
Wall Collisions   (+): 817
Adjancent Enemies (+): 133
Flanking Enemies  (+): 59
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 180, T_ALIVE = 10042
A_HP = 90162460, A_STAMINA = 116526800, A_DAMAGE = 22309530
A_HP/T_ALIVE = 8978.54, A_STAMINA/T_ALIVE = 11603.9, A_DAMAGE/T_ALIVE = 2221.62
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5020
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 583
Wall Collisions   (-): 361
Adjancent Enemies (-): 32
Flanking Enemies  (-): 190
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1306
Wall Collisions   (+): 1108
Adjancent Enemies (+): 156
Flanking Enemies  (+): 42
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 115, T_ALIVE = 5688
A_HP = 38403860, A_STAMINA = 71632340, A_DAMAGE = 8680495
A_HP/T_ALIVE = 6751.73, A_STAMINA/T_ALIVE = 12593.6, A_DAMAGE/T_ALIVE = 1526.11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2843
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 265
Wall Collisions   (-): 83
Adjancent Enemies (-): 26
Flanking Enemies  (-): 156
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 739
Wall Collisions   (+): 607
Adjancent Enemies (+): 110
Flanking Enemies  (+): 22
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 211, T_ALIVE = 11458
A_HP = 86504355, A_STAMINA = 127770380, A_DAMAGE = 28910520
A_HP/T_ALIVE = 7549.69, A_STAMINA/T_ALIVE = 11151.2, A_DAMAGE/T_ALIVE = 2523.17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5728
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 632
Wall Collisions   (-): 330
Adjancent Enemies (-): 41
Flanking Enemies  (-): 261
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1368
Wall Collisions   (+): 1149
Adjancent Enemies (+): 174
Flanking Enemies  (+): 45
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 123, T_ALIVE = 7688
A_HP = 77860730, A_STAMINA = 86007860, A_DAMAGE = 12644922
A_HP/T_ALIVE = 10127.6, A_STAMINA/T_ALIVE = 11187.3, A_DAMAGE/T_ALIVE = 1644.76
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3843
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 502
Wall Collisions   (-): 159
Adjancent Enemies (-): 172
Flanking Enemies  (-): 171
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1107
Wall Collisions   (+): 695
Adjancent Enemies (+): 362
Flanking Enemies  (+): 50
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 122, T_ALIVE = 6656
A_HP = 50996770, A_STAMINA = 85504100, A_DAMAGE = 16079290
A_HP/T_ALIVE = 7661.77, A_STAMINA/T_ALIVE = 12846.2, A_DAMAGE/T_ALIVE = 2415.76
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3327
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 444
Wall Collisions   (-): 247
Adjancent Enemies (-): 24
Flanking Enemies  (-): 173
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 920
Wall Collisions   (+): 790
Adjancent Enemies (+): 100
Flanking Enemies  (+): 30
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 249, T_ALIVE = 15110
A_HP = 131284200, A_STAMINA = 154951580, A_DAMAGE = 40693195
A_HP/T_ALIVE = 8688.56, A_STAMINA/T_ALIVE = 10254.9, A_DAMAGE/T_ALIVE = 2693.13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7554
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 693
Wall Collisions   (-): 296
Adjancent Enemies (-): 47
Flanking Enemies  (-): 350
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2569
Wall Collisions   (+): 2297
Adjancent Enemies (+): 206
Flanking Enemies  (+): 66
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 201, T_ALIVE = 12548
A_HP = 133887225, A_STAMINA = 136413380, A_DAMAGE = 26738951
A_HP/T_ALIVE = 10670, A_STAMINA/T_ALIVE = 10871.3, A_DAMAGE/T_ALIVE = 2130.93
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6273
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 722
Wall Collisions   (-): 297
Adjancent Enemies (-): 169
Flanking Enemies  (-): 256
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1643
Wall Collisions   (+): 1343
Adjancent Enemies (+): 260
Flanking Enemies  (+): 40
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 131, T_ALIVE = 7508
A_HP = 61656205, A_STAMINA = 92888660, A_DAMAGE = 12597600
A_HP/T_ALIVE = 8212.07, A_STAMINA/T_ALIVE = 12372, A_DAMAGE/T_ALIVE = 1677.89
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3753
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 348
Wall Collisions   (-): 166
Adjancent Enemies (-): 17
Flanking Enemies  (-): 165
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 949
Wall Collisions   (+): 800
Adjancent Enemies (+): 129
Flanking Enemies  (+): 20
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 168, T_ALIVE = 12682
A_HP = 94257100, A_STAMINA = 121938820, A_DAMAGE = 12117701
A_HP/T_ALIVE = 7432.35, A_STAMINA/T_ALIVE = 9615.11, A_DAMAGE/T_ALIVE = 955.504
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6340
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 796
Wall Collisions   (-): 434
Adjancent Enemies (-): 134
Flanking Enemies  (-): 228
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2172
Wall Collisions   (+): 1579
Adjancent Enemies (+): 548
Flanking Enemies  (+): 45
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 170, T_ALIVE = 11232
A_HP = 97255900, A_STAMINA = 132418160, A_DAMAGE = 25496975
A_HP/T_ALIVE = 8658.82, A_STAMINA/T_ALIVE = 11789.4, A_DAMAGE/T_ALIVE = 2270.03
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5615
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 664
Wall Collisions   (-): 413
Adjancent Enemies (-): 29
Flanking Enemies  (-): 222
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1763
Wall Collisions   (+): 1572
Adjancent Enemies (+): 147
Flanking Enemies  (+): 44
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 144, T_ALIVE = 8464
A_HP = 85035140, A_STAMINA = 106608340, A_DAMAGE = 17014225
A_HP/T_ALIVE = 10046.7, A_STAMINA/T_ALIVE = 12595.5, A_DAMAGE/T_ALIVE = 2010.19
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4231
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 465
Wall Collisions   (-): 258
Adjancent Enemies (-): 26
Flanking Enemies  (-): 181
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1388
Wall Collisions   (+): 1225
Adjancent Enemies (+): 122
Flanking Enemies  (+): 41
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 115, T_ALIVE = 7490
A_HP = 57604300, A_STAMINA = 83584380, A_DAMAGE = 11375219
A_HP/T_ALIVE = 7690.83, A_STAMINA/T_ALIVE = 11159.5, A_DAMAGE/T_ALIVE = 1518.72
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3744
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 545
Wall Collisions   (-): 135
Adjancent Enemies (-): 234
Flanking Enemies  (-): 176
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 968
Wall Collisions   (+): 627
Adjancent Enemies (+): 314
Flanking Enemies  (+): 27
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 101, T_ALIVE = 6150
A_HP = 42301110, A_STAMINA = 73183040, A_DAMAGE = 13059199
A_HP/T_ALIVE = 6878.23, A_STAMINA/T_ALIVE = 11899.7, A_DAMAGE/T_ALIVE = 2123.45
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3074
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 527
Wall Collisions   (-): 275
Adjancent Enemies (-): 105
Flanking Enemies  (-): 147
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 955
Wall Collisions   (+): 747
Adjancent Enemies (+): 188
Flanking Enemies  (+): 20
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 98, T_ALIVE = 4794
A_HP = 48858265, A_STAMINA = 63549380, A_DAMAGE = 12310395
A_HP/T_ALIVE = 10191.5, A_STAMINA/T_ALIVE = 13256, A_DAMAGE/T_ALIVE = 2567.88
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2396
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 275
Wall Collisions   (-): 127
Adjancent Enemies (-): 26
Flanking Enemies  (-): 122
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 557
Wall Collisions   (+): 464
Adjancent Enemies (+): 73
Flanking Enemies  (+): 20
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 85, T_ALIVE = 5412
A_HP = 50621385, A_STAMINA = 61542640, A_DAMAGE = 9478537
A_HP/T_ALIVE = 9353.54, A_STAMINA/T_ALIVE = 11371.5, A_DAMAGE/T_ALIVE = 1751.39
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2705
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 329
Wall Collisions   (-): 55
Adjancent Enemies (-): 165
Flanking Enemies  (-): 109
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 923
Wall Collisions   (+): 531
Adjancent Enemies (+): 375
Flanking Enemies  (+): 17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 185, T_ALIVE = 9708
A_HP = 65106125, A_STAMINA = 110819740, A_DAMAGE = 18702230
A_HP/T_ALIVE = 6706.44, A_STAMINA/T_ALIVE = 11415.3, A_DAMAGE/T_ALIVE = 1926.48
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4853
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 528
Wall Collisions   (-): 210
Adjancent Enemies (-): 47
Flanking Enemies  (-): 271
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1482
Wall Collisions   (+): 1267
Adjancent Enemies (+): 159
Flanking Enemies  (+): 56
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 284, T_ALIVE = 18928
A_HP = 160027360, A_STAMINA = 189744020, A_DAMAGE = 40357040
A_HP/T_ALIVE = 8454.53, A_STAMINA/T_ALIVE = 10024.5, A_DAMAGE/T_ALIVE = 2132.13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 9463
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1483
Wall Collisions   (-): 1081
Adjancent Enemies (-): 43
Flanking Enemies  (-): 359
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2684
Wall Collisions   (+): 2366
Adjancent Enemies (+): 262
Flanking Enemies  (+): 56
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 423, T_ALIVE = 24392
A_HP = 325979410, A_STAMINA = 175908800, A_DAMAGE = 70276615
A_HP/T_ALIVE = 13364.2, A_STAMINA/T_ALIVE = 7211.74, A_DAMAGE/T_ALIVE = 2881.13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 12195
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1543
Wall Collisions   (-): 912
Adjancent Enemies (-): 55
Flanking Enemies  (-): 576
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3783
Wall Collisions   (+): 3247
Adjancent Enemies (+): 476
Flanking Enemies  (+): 60
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

------------------

## backup-FT [MASK1]:

```
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 162, T_ALIVE = 8788
A_HP = 55440280, A_STAMINA = 101083620, A_DAMAGE = 24007895
A_HP/T_ALIVE = 6308.63, A_STAMINA/T_ALIVE = 11502.5, A_DAMAGE/T_ALIVE = 2731.9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4393
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 627
Wall Collisions   (-): 416
Adjancent Enemies (-): 24
Flanking Enemies  (-): 187
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1418
Wall Collisions   (+): 1237
Adjancent Enemies (+): 142
Flanking Enemies  (+): 39
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 226, T_ALIVE = 12744
A_HP = 69347230, A_STAMINA = 128874840, A_DAMAGE = 17438375
A_HP/T_ALIVE = 5441.56, A_STAMINA/T_ALIVE = 10112.6, A_DAMAGE/T_ALIVE = 1368.36
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6371
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 591
Wall Collisions   (-): 285
Adjancent Enemies (-): 43
Flanking Enemies  (-): 263
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1763
Wall Collisions   (+): 1471
Adjancent Enemies (+): 225
Flanking Enemies  (+): 67
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 112, T_ALIVE = 5934
A_HP = 41757495, A_STAMINA = 75486460, A_DAMAGE = 11019665
A_HP/T_ALIVE = 7036.99, A_STAMINA/T_ALIVE = 12721, A_DAMAGE/T_ALIVE = 1857.04
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2966
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 187
Wall Collisions   (-): 34
Adjancent Enemies (-): 18
Flanking Enemies  (-): 135
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 699
Wall Collisions   (+): 566
Adjancent Enemies (+): 101
Flanking Enemies  (+): 32
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 164, T_ALIVE = 8404
A_HP = 60909340, A_STAMINA = 95218720, A_DAMAGE = 12498625
A_HP/T_ALIVE = 7247.66, A_STAMINA/T_ALIVE = 11330.2, A_DAMAGE/T_ALIVE = 1487.22
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4201
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 428
Wall Collisions   (-): 156
Adjancent Enemies (-): 32
Flanking Enemies  (-): 240
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 949
Wall Collisions   (+): 745
Adjancent Enemies (+): 167
Flanking Enemies  (+): 37
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 417, T_ALIVE = 23552
A_HP = 274677990, A_STAMINA = 175911840, A_DAMAGE = 73229445
A_HP/T_ALIVE = 11662.6, A_STAMINA/T_ALIVE = 7469.08, A_DAMAGE/T_ALIVE = 3109.27
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 11775
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1288
Wall Collisions   (-): 670
Adjancent Enemies (-): 60
Flanking Enemies  (-): 558
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3485
Wall Collisions   (+): 3021
Adjancent Enemies (+): 385
Flanking Enemies  (+): 79
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 106, T_ALIVE = 7564
A_HP = 68418415, A_STAMINA = 86251100, A_DAMAGE = 11773959
A_HP/T_ALIVE = 9045.27, A_STAMINA/T_ALIVE = 11402.8, A_DAMAGE/T_ALIVE = 1556.58
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3781
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 830
Wall Collisions   (-): 499
Adjancent Enemies (-): 177
Flanking Enemies  (-): 154
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1291
Wall Collisions   (+): 1029
Adjancent Enemies (+): 218
Flanking Enemies  (+): 44
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 115, T_ALIVE = 5786
A_HP = 48616210, A_STAMINA = 73525120, A_DAMAGE = 16457200
A_HP/T_ALIVE = 8402.39, A_STAMINA/T_ALIVE = 12707.4, A_DAMAGE/T_ALIVE = 2844.31
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2892
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 372
Wall Collisions   (-): 185
Adjancent Enemies (-): 7
Flanking Enemies  (-): 180
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 807
Wall Collisions   (+): 694
Adjancent Enemies (+): 108
Flanking Enemies  (+): 5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 215, T_ALIVE = 10566
A_HP = 77049240, A_STAMINA = 117707580, A_DAMAGE = 24833670
A_HP/T_ALIVE = 7292.19, A_STAMINA/T_ALIVE = 11140.2, A_DAMAGE/T_ALIVE = 2350.34
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5282
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 809
Wall Collisions   (-): 459
Adjancent Enemies (-): 52
Flanking Enemies  (-): 298
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1381
Wall Collisions   (+): 1154
Adjancent Enemies (+): 178
Flanking Enemies  (+): 49
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 237, T_ALIVE = 13318
A_HP = 77571835, A_STAMINA = 144489840, A_DAMAGE = 23721620
A_HP/T_ALIVE = 5824.59, A_STAMINA/T_ALIVE = 10849.2, A_DAMAGE/T_ALIVE = 1781.17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6658
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 696
Wall Collisions   (-): 331
Adjancent Enemies (-): 40
Flanking Enemies  (-): 325
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2121
Wall Collisions   (+): 1846
Adjancent Enemies (+): 212
Flanking Enemies  (+): 63
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 213, T_ALIVE = 10780
A_HP = 70782270, A_STAMINA = 116846580, A_DAMAGE = 31270750
A_HP/T_ALIVE = 6566.07, A_STAMINA/T_ALIVE = 10839.2, A_DAMAGE/T_ALIVE = 2900.81
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5389
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 474
Wall Collisions   (-): 159
Adjancent Enemies (-): 31
Flanking Enemies  (-): 284
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1142
Wall Collisions   (+): 904
Adjancent Enemies (+): 188
Flanking Enemies  (+): 50
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 205, T_ALIVE = 11512
A_HP = 83741990, A_STAMINA = 131614380, A_DAMAGE = 24409555
A_HP/T_ALIVE = 7274.32, A_STAMINA/T_ALIVE = 11432.8, A_DAMAGE/T_ALIVE = 2120.36
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5755
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 850
Wall Collisions   (-): 536
Adjancent Enemies (-): 36
Flanking Enemies  (-): 278
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1585
Wall Collisions   (+): 1357
Adjancent Enemies (+): 176
Flanking Enemies  (+): 52
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 181, T_ALIVE = 8514
A_HP = 65330500, A_STAMINA = 100067220, A_DAMAGE = 19789900
A_HP/T_ALIVE = 7673.3, A_STAMINA/T_ALIVE = 11753.3, A_DAMAGE/T_ALIVE = 2324.4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4256
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 444
Wall Collisions   (-): 183
Adjancent Enemies (-): 34
Flanking Enemies  (-): 227
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1083
Wall Collisions   (+): 879
Adjancent Enemies (+): 153
Flanking Enemies  (+): 51
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 74, T_ALIVE = 3400
A_HP = 29121370, A_STAMINA = 46367180, A_DAMAGE = 8694515
A_HP/T_ALIVE = 8565.11, A_STAMINA/T_ALIVE = 13637.4, A_DAMAGE/T_ALIVE = 2557.21
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 1699
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 145
Wall Collisions   (-): 22
Adjancent Enemies (-): 14
Flanking Enemies  (-): 109
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 440
Wall Collisions   (+): 355
Adjancent Enemies (+): 61
Flanking Enemies  (+): 24
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 182, T_ALIVE = 12924
A_HP = 70357445, A_STAMINA = 142340680, A_DAMAGE = 25072720
A_HP/T_ALIVE = 5443.94, A_STAMINA/T_ALIVE = 11013.7, A_DAMAGE/T_ALIVE = 1940.01
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6461
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1180
Wall Collisions   (-): 903
Adjancent Enemies (-): 36
Flanking Enemies  (-): 241
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1808
Wall Collisions   (+): 1603
Adjancent Enemies (+): 164
Flanking Enemies  (+): 41
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 138, T_ALIVE = 7716
A_HP = 60779930, A_STAMINA = 95451700, A_DAMAGE = 16549075
A_HP/T_ALIVE = 7877.13, A_STAMINA/T_ALIVE = 12370.6, A_DAMAGE/T_ALIVE = 2144.77
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3857
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 414
Wall Collisions   (-): 185
Adjancent Enemies (-): 28
Flanking Enemies  (-): 201
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 975
Wall Collisions   (+): 815
Adjancent Enemies (+): 112
Flanking Enemies  (+): 48
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 144, T_ALIVE = 9254
A_HP = 73881815, A_STAMINA = 111588800, A_DAMAGE = 13777307
A_HP/T_ALIVE = 7983.77, A_STAMINA/T_ALIVE = 12058.4, A_DAMAGE/T_ALIVE = 1488.79
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4626
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 373
Wall Collisions   (-): 138
Adjancent Enemies (-): 38
Flanking Enemies  (-): 197
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1574
Wall Collisions   (+): 1348
Adjancent Enemies (+): 196
Flanking Enemies  (+): 30
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 230, T_ALIVE = 13118
A_HP = 106746100, A_STAMINA = 139818360, A_DAMAGE = 29838840
A_HP/T_ALIVE = 8137.38, A_STAMINA/T_ALIVE = 10658.5, A_DAMAGE/T_ALIVE = 2274.65
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6558
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 705
Wall Collisions   (-): 373
Adjancent Enemies (-): 36
Flanking Enemies  (-): 296
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1796
Wall Collisions   (+): 1541
Adjancent Enemies (+): 202
Flanking Enemies  (+): 53
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 216, T_ALIVE = 14642
A_HP = 101172675, A_STAMINA = 155040920, A_DAMAGE = 29986842
A_HP/T_ALIVE = 6909.76, A_STAMINA/T_ALIVE = 10588.8, A_DAMAGE/T_ALIVE = 2048
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7320
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1253
Wall Collisions   (-): 782
Adjancent Enemies (-): 180
Flanking Enemies  (-): 291
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2379
Wall Collisions   (+): 1983
Adjancent Enemies (+): 328
Flanking Enemies  (+): 68
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 91, T_ALIVE = 6682
A_HP = 64768960, A_STAMINA = 78065500, A_DAMAGE = 10013260
A_HP/T_ALIVE = 9693.05, A_STAMINA/T_ALIVE = 11683, A_DAMAGE/T_ALIVE = 1498.54
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3340
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 410
Wall Collisions   (-): 91
Adjancent Enemies (-): 157
Flanking Enemies  (-): 162
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 876
Wall Collisions   (+): 547
Adjancent Enemies (+): 303
Flanking Enemies  (+): 26
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 147, T_ALIVE = 8414
A_HP = 90813550, A_STAMINA = 101930420, A_DAMAGE = 17165295
A_HP/T_ALIVE = 10793.1, A_STAMINA/T_ALIVE = 12114.4, A_DAMAGE/T_ALIVE = 2040.09
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4206
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 406
Wall Collisions   (-): 207
Adjancent Enemies (-): 39
Flanking Enemies  (-): 160
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1318
Wall Collisions   (+): 1151
Adjancent Enemies (+): 122
Flanking Enemies  (+): 45
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 116, T_ALIVE = 6632
A_HP = 53288495, A_STAMINA = 84606900, A_DAMAGE = 15671535
A_HP/T_ALIVE = 8035.06, A_STAMINA/T_ALIVE = 12757.4, A_DAMAGE/T_ALIVE = 2363.02
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3315
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 263
Wall Collisions   (-): 108
Adjancent Enemies (-): 24
Flanking Enemies  (-): 131
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 667
Wall Collisions   (+): 546
Adjancent Enemies (+): 94
Flanking Enemies  (+): 27
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 152, T_ALIVE = 7652
A_HP = 56202065, A_STAMINA = 93237780, A_DAMAGE = 18794740
A_HP/T_ALIVE = 7344.75, A_STAMINA/T_ALIVE = 12184.8, A_DAMAGE/T_ALIVE = 2456.19
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3825
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 308
Wall Collisions   (-): 77
Adjancent Enemies (-): 34
Flanking Enemies  (-): 197
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 894
Wall Collisions   (+): 734
Adjancent Enemies (+): 120
Flanking Enemies  (+): 40
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 59, T_ALIVE = 4504
A_HP = 41509495, A_STAMINA = 39256480, A_DAMAGE = 5246998
A_HP/T_ALIVE = 9216.14, A_STAMINA/T_ALIVE = 8715.91, A_DAMAGE/T_ALIVE = 1164.96
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2251
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 231
Wall Collisions   (-): 31
Adjancent Enemies (-): 105
Flanking Enemies  (-): 95
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 993
Wall Collisions   (+): 224
Adjancent Enemies (+): 758
Flanking Enemies  (+): 11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 267, T_ALIVE = 12782
A_HP = 102789100, A_STAMINA = 128184200, A_DAMAGE = 32704415
A_HP/T_ALIVE = 8041.71, A_STAMINA/T_ALIVE = 10028.5, A_DAMAGE/T_ALIVE = 2558.63
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6390
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 562
Wall Collisions   (-): 176
Adjancent Enemies (-): 38
Flanking Enemies  (-): 348
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1609
Wall Collisions   (+): 1334
Adjancent Enemies (+): 237
Flanking Enemies  (+): 38
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 256, T_ALIVE = 15244
A_HP = 107310060, A_STAMINA = 161976560, A_DAMAGE = 42619045
A_HP/T_ALIVE = 7039.49, A_STAMINA/T_ALIVE = 10625.6, A_DAMAGE/T_ALIVE = 2795.79
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7621
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1056
Wall Collisions   (-): 709
Adjancent Enemies (-): 42
Flanking Enemies  (-): 305
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2213
Wall Collisions   (+): 1958
Adjancent Enemies (+): 218
Flanking Enemies  (+): 37
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 76, T_ALIVE = 3774
A_HP = 32025610, A_STAMINA = 50670500, A_DAMAGE = 8397605
A_HP/T_ALIVE = 8485.85, A_STAMINA/T_ALIVE = 13426.2, A_DAMAGE/T_ALIVE = 2225.12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 1886
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 196
Wall Collisions   (-): 61
Adjancent Enemies (-): 14
Flanking Enemies  (-): 121
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 484
Wall Collisions   (+): 390
Adjancent Enemies (+): 68
Flanking Enemies  (+): 26
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 126, T_ALIVE = 6124
A_HP = 50815625, A_STAMINA = 78290940, A_DAMAGE = 12127060
A_HP/T_ALIVE = 8297.78, A_STAMINA/T_ALIVE = 12784.3, A_DAMAGE/T_ALIVE = 1980.25
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3061
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 255
Wall Collisions   (-): 60
Adjancent Enemies (-): 27
Flanking Enemies  (-): 168
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 766
Wall Collisions   (+): 623
Adjancent Enemies (+): 107
Flanking Enemies  (+): 36
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 293, T_ALIVE = 14480
A_HP = 120172200, A_STAMINA = 134062680, A_DAMAGE = 28456165
A_HP/T_ALIVE = 8299.19, A_STAMINA/T_ALIVE = 9258.47, A_DAMAGE/T_ALIVE = 1965.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7239
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 710
Wall Collisions   (-): 270
Adjancent Enemies (-): 56
Flanking Enemies  (-): 384
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2108
Wall Collisions   (+): 1757
Adjancent Enemies (+): 284
Flanking Enemies  (+): 67
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 211, T_ALIVE = 11282
A_HP = 103865065, A_STAMINA = 119609160, A_DAMAGE = 20787425
A_HP/T_ALIVE = 9206.26, A_STAMINA/T_ALIVE = 10601.8, A_DAMAGE/T_ALIVE = 1842.53
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5640
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 700
Wall Collisions   (-): 360
Adjancent Enemies (-): 64
Flanking Enemies  (-): 276
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1499
Wall Collisions   (+): 1246
Adjancent Enemies (+): 198
Flanking Enemies  (+): 55
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 116, T_ALIVE = 6016
A_HP = 57908205, A_STAMINA = 77488920, A_DAMAGE = 12031875
A_HP/T_ALIVE = 9625.7, A_STAMINA/T_ALIVE = 12880.5, A_DAMAGE/T_ALIVE = 1999.98
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3007
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 393
Wall Collisions   (-): 237
Adjancent Enemies (-): 18
Flanking Enemies  (-): 138
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 804
Wall Collisions   (+): 651
Adjancent Enemies (+): 120
Flanking Enemies  (+): 33
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 89, T_ALIVE = 4240
A_HP = 37143325, A_STAMINA = 55295360, A_DAMAGE = 6760645
A_HP/T_ALIVE = 8760.22, A_STAMINA/T_ALIVE = 13041.4, A_DAMAGE/T_ALIVE = 1594.49
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2119
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 190
Wall Collisions   (-): 42
Adjancent Enemies (-): 31
Flanking Enemies  (-): 117
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 441
Wall Collisions   (+): 323
Adjancent Enemies (+): 83
Flanking Enemies  (+): 35
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 60, T_ALIVE = 3524
A_HP = 28563845, A_STAMINA = 47636720, A_DAMAGE = 4652721
A_HP/T_ALIVE = 8105.52, A_STAMINA/T_ALIVE = 13517.8, A_DAMAGE/T_ALIVE = 1320.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 1761
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 151
Wall Collisions   (-): 27
Adjancent Enemies (-): 46
Flanking Enemies  (-): 78
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 369
Wall Collisions   (+): 268
Adjancent Enemies (+): 77
Flanking Enemies  (+): 24
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 192, T_ALIVE = 11528
A_HP = 120783270, A_STAMINA = 130710180, A_DAMAGE = 20039253
A_HP/T_ALIVE = 10477.4, A_STAMINA/T_ALIVE = 11338.5, A_DAMAGE/T_ALIVE = 1738.31
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5763
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 632
Wall Collisions   (-): 308
Adjancent Enemies (-): 88
Flanking Enemies  (-): 236
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1805
Wall Collisions   (+): 1430
Adjancent Enemies (+): 339
Flanking Enemies  (+): 36
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 123, T_ALIVE = 8606
A_HP = 75664555, A_STAMINA = 96308160, A_DAMAGE = 13646747
A_HP/T_ALIVE = 8792.07, A_STAMINA/T_ALIVE = 11190.8, A_DAMAGE/T_ALIVE = 1585.72
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4302
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 440
Wall Collisions   (-): 93
Adjancent Enemies (-): 167
Flanking Enemies  (-): 180
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1366
Wall Collisions   (+): 977
Adjancent Enemies (+): 359
Flanking Enemies  (+): 30
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 94, T_ALIVE = 4536
A_HP = 40220000, A_STAMINA = 60744440, A_DAMAGE = 9801670
A_HP/T_ALIVE = 8866.84, A_STAMINA/T_ALIVE = 13391.6, A_DAMAGE/T_ALIVE = 2160.86
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2267
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 174
Wall Collisions   (-): 30
Adjancent Enemies (-): 21
Flanking Enemies  (-): 123
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 428
Wall Collisions   (+): 325
Adjancent Enemies (+): 79
Flanking Enemies  (+): 24
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 166, T_ALIVE = 8296
A_HP = 75343355, A_STAMINA = 100768020, A_DAMAGE = 21780410
A_HP/T_ALIVE = 9081.89, A_STAMINA/T_ALIVE = 12146.6, A_DAMAGE/T_ALIVE = 2625.41
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4147
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 389
Wall Collisions   (-): 102
Adjancent Enemies (-): 30
Flanking Enemies  (-): 257
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1102
Wall Collisions   (+): 916
Adjancent Enemies (+): 137
Flanking Enemies  (+): 49
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 95, T_ALIVE = 6628
A_HP = 44248525, A_STAMINA = 86285120, A_DAMAGE = 12404185
A_HP/T_ALIVE = 6676, A_STAMINA/T_ALIVE = 13018.3, A_DAMAGE/T_ALIVE = 1871.48
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3313
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 522
Wall Collisions   (-): 371
Adjancent Enemies (-): 19
Flanking Enemies  (-): 132
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1029
Wall Collisions   (+): 920
Adjancent Enemies (+): 84
Flanking Enemies  (+): 25
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 384, T_ALIVE = 21978
A_HP = 171618690, A_STAMINA = 175905840, A_DAMAGE = 53613645
A_HP/T_ALIVE = 7808.66, A_STAMINA/T_ALIVE = 8003.72, A_DAMAGE/T_ALIVE = 2439.42
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 10988
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1418
Wall Collisions   (-): 778
Adjancent Enemies (-): 62
Flanking Enemies  (-): 578
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3553
Wall Collisions   (+): 3139
Adjancent Enemies (+): 331
Flanking Enemies  (+): 83
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 82, T_ALIVE = 5642
A_HP = 43785600, A_STAMINA = 64397540, A_DAMAGE = 7568659
A_HP/T_ALIVE = 7760.65, A_STAMINA/T_ALIVE = 11414, A_DAMAGE/T_ALIVE = 1341.49
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2820
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 285
Wall Collisions   (-): 37
Adjancent Enemies (-): 140
Flanking Enemies  (-): 108
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1041
Wall Collisions   (+): 592
Adjancent Enemies (+): 420
Flanking Enemies  (+): 29
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 151, T_ALIVE = 10116
A_HP = 84946115, A_STAMINA = 121148460, A_DAMAGE = 18075205
A_HP/T_ALIVE = 8397.2, A_STAMINA/T_ALIVE = 11975.9, A_DAMAGE/T_ALIVE = 1786.79
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5057
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 899
Wall Collisions   (-): 687
Adjancent Enemies (-): 32
Flanking Enemies  (-): 180
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1583
Wall Collisions   (+): 1414
Adjancent Enemies (+): 133
Flanking Enemies  (+): 36
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 192, T_ALIVE = 10678
A_HP = 100266780, A_STAMINA = 127527100, A_DAMAGE = 27250680
A_HP/T_ALIVE = 9390.03, A_STAMINA/T_ALIVE = 11943, A_DAMAGE/T_ALIVE = 2552.04
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5338
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 483
Wall Collisions   (-): 200
Adjancent Enemies (-): 33
Flanking Enemies  (-): 250
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1736
Wall Collisions   (+): 1519
Adjancent Enemies (+): 161
Flanking Enemies  (+): 56
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 118, T_ALIVE = 6022
A_HP = 41021100, A_STAMINA = 74831420, A_DAMAGE = 10413895
A_HP/T_ALIVE = 6811.87, A_STAMINA/T_ALIVE = 12426.3, A_DAMAGE/T_ALIVE = 1729.31
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3010
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 240
Wall Collisions   (-): 69
Adjancent Enemies (-): 28
Flanking Enemies  (-): 143
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 805
Wall Collisions   (+): 665
Adjancent Enemies (+): 109
Flanking Enemies  (+): 31
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 117, T_ALIVE = 7018
A_HP = 50592070, A_STAMINA = 89370860, A_DAMAGE = 15375075
A_HP/T_ALIVE = 7208.9, A_STAMINA/T_ALIVE = 12734.5, A_DAMAGE/T_ALIVE = 2190.81
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3508
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 244
Wall Collisions   (-): 72
Adjancent Enemies (-): 15
Flanking Enemies  (-): 157
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 779
Wall Collisions   (+): 649
Adjancent Enemies (+): 108
Flanking Enemies  (+): 22
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 113, T_ALIVE = 6788
A_HP = 62800985, A_STAMINA = 78692160, A_DAMAGE = 9971602
A_HP/T_ALIVE = 9251.77, A_STAMINA/T_ALIVE = 11592.8, A_DAMAGE/T_ALIVE = 1469
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3393
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 467
Wall Collisions   (-): 218
Adjancent Enemies (-): 106
Flanking Enemies  (-): 143
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 990
Wall Collisions   (+): 662
Adjancent Enemies (+): 296
Flanking Enemies  (+): 32
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 247, T_ALIVE = 14416
A_HP = 91893005, A_STAMINA = 136341920, A_DAMAGE = 22466759
A_HP/T_ALIVE = 6374.38, A_STAMINA/T_ALIVE = 9457.68, A_DAMAGE/T_ALIVE = 1558.46
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7207
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1048
Wall Collisions   (-): 446
Adjancent Enemies (-): 271
Flanking Enemies  (-): 331
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2173
Wall Collisions   (+): 1752
Adjancent Enemies (+): 345
Flanking Enemies  (+): 76
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 165, T_ALIVE = 8266
A_HP = 62963530, A_STAMINA = 97977240, A_DAMAGE = 20907955
A_HP/T_ALIVE = 7617.17, A_STAMINA/T_ALIVE = 11853, A_DAMAGE/T_ALIVE = 2529.39
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4132
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 393
Wall Collisions   (-): 146
Adjancent Enemies (-): 40
Flanking Enemies  (-): 207
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1009
Wall Collisions   (+): 817
Adjancent Enemies (+): 133
Flanking Enemies  (+): 59
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 180, T_ALIVE = 10042
A_HP = 90162460, A_STAMINA = 116526800, A_DAMAGE = 22309530
A_HP/T_ALIVE = 8978.54, A_STAMINA/T_ALIVE = 11603.9, A_DAMAGE/T_ALIVE = 2221.62
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5020
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 583
Wall Collisions   (-): 361
Adjancent Enemies (-): 32
Flanking Enemies  (-): 190
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1306
Wall Collisions   (+): 1108
Adjancent Enemies (+): 156
Flanking Enemies  (+): 42
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 115, T_ALIVE = 5688
A_HP = 38403860, A_STAMINA = 71632340, A_DAMAGE = 8680495
A_HP/T_ALIVE = 6751.73, A_STAMINA/T_ALIVE = 12593.6, A_DAMAGE/T_ALIVE = 1526.11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2843
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 265
Wall Collisions   (-): 83
Adjancent Enemies (-): 26
Flanking Enemies  (-): 156
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 739
Wall Collisions   (+): 607
Adjancent Enemies (+): 110
Flanking Enemies  (+): 22
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 211, T_ALIVE = 11458
A_HP = 86504355, A_STAMINA = 127770380, A_DAMAGE = 28910520
A_HP/T_ALIVE = 7549.69, A_STAMINA/T_ALIVE = 11151.2, A_DAMAGE/T_ALIVE = 2523.17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5728
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 632
Wall Collisions   (-): 330
Adjancent Enemies (-): 41
Flanking Enemies  (-): 261
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1368
Wall Collisions   (+): 1149
Adjancent Enemies (+): 174
Flanking Enemies  (+): 45
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 123, T_ALIVE = 7688
A_HP = 77860730, A_STAMINA = 86007860, A_DAMAGE = 12644922
A_HP/T_ALIVE = 10127.6, A_STAMINA/T_ALIVE = 11187.3, A_DAMAGE/T_ALIVE = 1644.76
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3843
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 502
Wall Collisions   (-): 159
Adjancent Enemies (-): 172
Flanking Enemies  (-): 171
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1107
Wall Collisions   (+): 695
Adjancent Enemies (+): 362
Flanking Enemies  (+): 50
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 122, T_ALIVE = 6656
A_HP = 50996770, A_STAMINA = 85504100, A_DAMAGE = 16079290
A_HP/T_ALIVE = 7661.77, A_STAMINA/T_ALIVE = 12846.2, A_DAMAGE/T_ALIVE = 2415.76
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3327
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 444
Wall Collisions   (-): 247
Adjancent Enemies (-): 24
Flanking Enemies  (-): 173
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 920
Wall Collisions   (+): 790
Adjancent Enemies (+): 100
Flanking Enemies  (+): 30
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 249, T_ALIVE = 15110
A_HP = 131284200, A_STAMINA = 154951580, A_DAMAGE = 40693195
A_HP/T_ALIVE = 8688.56, A_STAMINA/T_ALIVE = 10254.9, A_DAMAGE/T_ALIVE = 2693.13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 7554
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 693
Wall Collisions   (-): 296
Adjancent Enemies (-): 47
Flanking Enemies  (-): 350
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2569
Wall Collisions   (+): 2297
Adjancent Enemies (+): 206
Flanking Enemies  (+): 66
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 201, T_ALIVE = 12548
A_HP = 133887225, A_STAMINA = 136413380, A_DAMAGE = 26738951
A_HP/T_ALIVE = 10670, A_STAMINA/T_ALIVE = 10871.3, A_DAMAGE/T_ALIVE = 2130.93
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6273
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 722
Wall Collisions   (-): 297
Adjancent Enemies (-): 169
Flanking Enemies  (-): 256
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1643
Wall Collisions   (+): 1343
Adjancent Enemies (+): 260
Flanking Enemies  (+): 40
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 131, T_ALIVE = 7508
A_HP = 61656205, A_STAMINA = 92888660, A_DAMAGE = 12597600
A_HP/T_ALIVE = 8212.07, A_STAMINA/T_ALIVE = 12372, A_DAMAGE/T_ALIVE = 1677.89
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3753
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 348
Wall Collisions   (-): 166
Adjancent Enemies (-): 17
Flanking Enemies  (-): 165
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 949
Wall Collisions   (+): 800
Adjancent Enemies (+): 129
Flanking Enemies  (+): 20
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 168, T_ALIVE = 12682
A_HP = 94257100, A_STAMINA = 121938820, A_DAMAGE = 12117701
A_HP/T_ALIVE = 7432.35, A_STAMINA/T_ALIVE = 9615.11, A_DAMAGE/T_ALIVE = 955.504
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 6340
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 796
Wall Collisions   (-): 434
Adjancent Enemies (-): 134
Flanking Enemies  (-): 228
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2172
Wall Collisions   (+): 1579
Adjancent Enemies (+): 548
Flanking Enemies  (+): 45
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 170, T_ALIVE = 11232
A_HP = 97255900, A_STAMINA = 132418160, A_DAMAGE = 25496975
A_HP/T_ALIVE = 8658.82, A_STAMINA/T_ALIVE = 11789.4, A_DAMAGE/T_ALIVE = 2270.03
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 5615
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 664
Wall Collisions   (-): 413
Adjancent Enemies (-): 29
Flanking Enemies  (-): 222
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1763
Wall Collisions   (+): 1572
Adjancent Enemies (+): 147
Flanking Enemies  (+): 44
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 144, T_ALIVE = 8464
A_HP = 85035140, A_STAMINA = 106608340, A_DAMAGE = 17014225
A_HP/T_ALIVE = 10046.7, A_STAMINA/T_ALIVE = 12595.5, A_DAMAGE/T_ALIVE = 2010.19
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4231
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 465
Wall Collisions   (-): 258
Adjancent Enemies (-): 26
Flanking Enemies  (-): 181
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1388
Wall Collisions   (+): 1225
Adjancent Enemies (+): 122
Flanking Enemies  (+): 41
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 115, T_ALIVE = 7490
A_HP = 57604300, A_STAMINA = 83584380, A_DAMAGE = 11375219
A_HP/T_ALIVE = 7690.83, A_STAMINA/T_ALIVE = 11159.5, A_DAMAGE/T_ALIVE = 1518.72
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3744
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 545
Wall Collisions   (-): 135
Adjancent Enemies (-): 234
Flanking Enemies  (-): 176
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 968
Wall Collisions   (+): 627
Adjancent Enemies (+): 314
Flanking Enemies  (+): 27
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 101, T_ALIVE = 6150
A_HP = 42301110, A_STAMINA = 73183040, A_DAMAGE = 13059199
A_HP/T_ALIVE = 6878.23, A_STAMINA/T_ALIVE = 11899.7, A_DAMAGE/T_ALIVE = 2123.45
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 3074
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 527
Wall Collisions   (-): 275
Adjancent Enemies (-): 105
Flanking Enemies  (-): 147
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 955
Wall Collisions   (+): 747
Adjancent Enemies (+): 188
Flanking Enemies  (+): 20
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 98, T_ALIVE = 4794
A_HP = 48858265, A_STAMINA = 63549380, A_DAMAGE = 12310395
A_HP/T_ALIVE = 10191.5, A_STAMINA/T_ALIVE = 13256, A_DAMAGE/T_ALIVE = 2567.88
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2396
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 275
Wall Collisions   (-): 127
Adjancent Enemies (-): 26
Flanking Enemies  (-): 122
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 557
Wall Collisions   (+): 464
Adjancent Enemies (+): 73
Flanking Enemies  (+): 20
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 85, T_ALIVE = 5412
A_HP = 50621385, A_STAMINA = 61542640, A_DAMAGE = 9478537
A_HP/T_ALIVE = 9353.54, A_STAMINA/T_ALIVE = 11371.5, A_DAMAGE/T_ALIVE = 1751.39
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 2705
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 329
Wall Collisions   (-): 55
Adjancent Enemies (-): 165
Flanking Enemies  (-): 109
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 923
Wall Collisions   (+): 531
Adjancent Enemies (+): 375
Flanking Enemies  (+): 17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 185, T_ALIVE = 9708
A_HP = 65106125, A_STAMINA = 110819740, A_DAMAGE = 18702230
A_HP/T_ALIVE = 6706.44, A_STAMINA/T_ALIVE = 11415.3, A_DAMAGE/T_ALIVE = 1926.48
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 4853
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 528
Wall Collisions   (-): 210
Adjancent Enemies (-): 47
Flanking Enemies  (-): 271
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 1482
Wall Collisions   (+): 1267
Adjancent Enemies (+): 159
Flanking Enemies  (+): 56
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 284, T_ALIVE = 18928
A_HP = 160027360, A_STAMINA = 189744020, A_DAMAGE = 40357040
A_HP/T_ALIVE = 8454.53, A_STAMINA/T_ALIVE = 10024.5, A_DAMAGE/T_ALIVE = 2132.13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 9463
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1483
Wall Collisions   (-): 1081
Adjancent Enemies (-): 43
Flanking Enemies  (-): 359
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 2684
Wall Collisions   (+): 2366
Adjancent Enemies (+): 262
Flanking Enemies  (+): 56
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=========
Agent parameters: 3382220
Inference mode : 1
Data gathering : 0
T              : 1054
Model path     : bots/bot-bc-fap/backup-FT/model.pt
Dataset dir    : bots/bot-bc-fap/dataset/data_train
Next episode   : 28
================ METRICS ================
KILLS = 423, T_ALIVE = 24392
A_HP = 325979410, A_STAMINA = 175908800, A_DAMAGE = 70276615
A_HP/T_ALIVE = 13364.2, A_STAMINA/T_ALIVE = 7211.74, A_DAMAGE/T_ALIVE = 2881.13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total Actions        : 12195
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Failures     : 1543
Wall Collisions   (-): 912
Adjancent Enemies (-): 55
Flanking Enemies  (-): 576
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overall Success      : 3783
Wall Collisions   (+): 3247
Adjancent Enemies (+): 476
Flanking Enemies  (+): 60
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```