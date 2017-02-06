DQN-in-the-Caffe
================

DQN-in-the-Caffe is an implementation of Deep Q-Network using Caffe. 

Updates
--
- Replace ALE with xitari, the same emulator as dqn in torch.
- Update caffe with a "relative new" version.

How to Run
--
- Compile xitari
```
cd xitari
make
```

- Compile caffe
```
cd caffe
make all
```

- Compile dqn
```
cmake .
make
```
