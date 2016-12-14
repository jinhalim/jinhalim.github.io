import tensorflow as tf
# tensorflow를 tf라는 이름으로 불러온다.
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)
# 필요한 데이터 다운로드


# Set up model
x = tf.placeholder(tf.float32, [None, 784])
# x : 인풋레이어에 사용할 부정소숫점으로 된 변수 정의(784차원의 벡터로 단조화)
W = tf.Variable(tf.zeros([784, 10]))
# w : 784 X 10 개의 초기값 0을 갖는 벡터의 증거 생성
b = tf.Variable(tf.zeros([10]))
# b : 10개짜리 배열 생성
y = tf.nn.softmax(tf.matmul(x, W) + b)
# y = x * w + b
# x (784) * w(784*10) = x*w(10)
# x*w(10) + b(10) = y(10)
y_ = tf.placeholder(tf.float32, [None, 10])
# y_ : 아웃풋레이어에 사용할 부정소숫점으로 된 변수 정의(10차원의 벡터로 단조화)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# cross_entropy:교차 엔트로피 (-시그마y'log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# train_step : TensorFlow에게 학습도를 0.01로 준 경사 하강법(gradient descent) 알고리즘을 이용해 교차 엔트로피를 최소화 명령

# Session
init = tf.initialize_all_variables()
# 만든 변수들을 초기화

sess = tf.Session()
# 세션에서 모델을 시작하고 변수들을 초기화
sess.run(init)
# 실행

# Learning
# 학습을 1000번
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # 학습 세트로에서 100개의 무작위 데이터들의 일괄 처리(batch)를 가져옴
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  # placeholders를 대체하기 위한 일괄 처리 데이터에 train_step 피딩 실행

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# tf.equal 을 이용해 예측이 실제와 맞았는지 확인 가능
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# (많은 비율로 맞았는지 확인용) 부정소숫점으로 캐스팅한 후 평균값을 구함. true:1, false:0

# Result should be approximately 91%.
# 테스트 데이터를 대상 정확도 확인 
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))