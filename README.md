# AAE-exp
Adversarial Autoencoder 논문의 2-3 챕터 잠재벡터 분포조정 실험을 코드로 구현하고 실험하였다.

## 2-3 실험 결과     
![image](https://github.com/paokimsiwoong/AAE-exp/assets/37607763/2c1d0261-a29d-4485-ae97-f153b512ff77)

위 그림과 같은 구조로 encoder와 decoder, discriminator들을 구성하였고 논문과 동일하게 hidden unit이 1000개인 두개의 레이어를 사용하였다.

활성함수 ReLU를 적용하기 전에 Batch normalization을 적용하였고 dropout은 사용하지 않았다. 

encoder는 2차원 좌표 (x,y)와 표준편차 std ($\sigma_x,\sigma_y$)를 출력하였고 표준편차는 표준편차의 범위를 만족하기 위해 ReLU를 사용하였다. 

이 출력된 (x,y)와 std ($\sigma_x,\sigma_y$)는 $\mathcal{N}((0,0),I)$에서 추출한 Gaussian noise를 affine transformation할 때 사용하였다((x,y)로 translation과 rotation, std ($\sigma_x,\sigma_y$)는 scaling factor).

batch size는 100으로 진행하고 총 1000 epochs 학습을 진행하였다. 

optimizer로는 모두 SGD를 사용하고, Simultaneous gradient ascent-descent 대신 Alternating gradient ascent-descent를 사용해 optimizer마다 순방향, 역방향 전파를 한번씩 진행하였다. reconstruction cost의 learning rate는 0.01을 사용하고 나머지는 0.1을 사용하였다. learning rate scheduler로는 모두 MultiStelLR을 사용하였고, 50 epoch, 500 epoch에 learning rate를 1/10배 하였다.




![image1](https://github.com/paokimsiwoong/AAE-exp/assets/37607763/650ca6dc-d1ba-4084-8a5e-6d102965e9b1)

학습이 끝난 encoder가 MNIST의 1만개의 test set 데이터들을 2차원에 encode 한 결과

![image2](https://github.com/paokimsiwoong/AAE-exp/assets/37607763/5088facc-dbb5-4597-b759-264bc1190e56)

각각의 leaf에서 추출한 sample을 decoder에 입력해 생성한 결과

![image3](https://github.com/paokimsiwoong/AAE-exp/assets/37607763/7f4230a9-18e9-48ab-8078-031a42e3917e)

auto encoder의 복원 이미지와 원본이미지의 비교

![image4](https://github.com/paokimsiwoong/AAE-exp/assets/37607763/e4f7f62a-f70f-4cf1-bf45-d40e1cc9e770)
![image5](https://github.com/paokimsiwoong/AAE-exp/assets/37607763/f1aa2854-077a-4aca-a607-9b58fa303e9f)
![image6](https://github.com/paokimsiwoong/AAE-exp/assets/37607763/8003dbd7-f1ec-4772-9fca-0a64fcddc768)

## 참고문헌
[Adversarial Autoencoders](https://arxiv.org/abs/1511.05644)

[심층 신경망의 수학적 기초 ch5](https://www.math.snu.ac.kr/~ernestryu/courses/DNN/chapter5.pdf)
