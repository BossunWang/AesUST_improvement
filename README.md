# AesUST_improvement
original repository from https://github.com/EndyWon/AesUST

# training record
* v1 using original setting, but got artifact and repetitive pattern
* v2 adjust gan_weight smaller and reduce to occur artifact
* v3 training with content encoder and get the best content fidelity and hold another

# Evaluation

| version | content fidelity | global effect |    local patterns     |
|:-------:| :---------: | :---------------: |:---------------------:|
|   v1    |     0.606484     |       0.810894        |       0.599884        |
|   v2    |     0.596819     |        0.864735        |       0.594486        |
|   v3    |     0.639204     |        0.863310        |       0.584813        |
|   v4    |     0.640776     |        0.830611        |       0.586811        |
|   v5    |     0.622983     |        0.845234        |       0.591902        |
|   v6    |     0.612933     |        0.835005        |       0.590082        |
