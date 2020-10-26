import tensorflow as tf
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt


# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
# 하이퍼파라미터를 튜닝하기 위한 용도(흔들리면 무엇때문에 좋아졌는지 알기 어려움)
tf.set_random_seed(777)

def MinMaxScaler(data):
    # 데이터 모든숫자들을 최소 값만큼 뺀다.
    numerator = data - np.min(data, 0)
    # 최대값과 최소 값의 차이(A)를 구한다
    denominator = np.max(data, 0) - np.min(data, 0)
    # 너무 큰 값이 나오지 않도록 나눈다
    return numerator / (denominator + 1e-7)


# 하이퍼파라미터
seq_length = 7       # 1개 시퀀스의 길이(시계열데이터 입력 개수)
data_dim = 5         # Variable 개수
hidden_dim = 10      # 각 셀의 출력 크기
output_dim = 1       # 결과 분류 총 수
learning_rate = 0.01 # 학습률
epoch_num = 500      # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)


# 데이터를 로딩한다.
# 시작가, 고가, 저가, 거래량, 종가
xy = np.loadtxt('./stock_daily_price.csv', delimiter=',')

# 데이터 전처리
xy = xy[::-1] # 제일앞이 뒤로, 제일뒤가 앞으로 순서를 뒤집는다.
print("xy[0][0]: ", xy[0][0])
xy = MinMaxScaler(xy)
print("xy[0][0]: ", xy[0][0])
x = xy
y = xy[:, [-1]] # 마지막 열이 정답(주식 종가)이다.
print("x[0]: ", x[0])
print("y[0]: ",y[0])

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i : i+seq_length]
    _y = y[i + seq_length] # 다음 나타날 주가(정답)
    if i is 0:
        print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)


# 학습용/테스트용 데이터 생성
# 70%를 학습용 데이터로 사용
train_size = int(len(dataY) * 0.7)
# 나머지(30%)를 테스트용 데이터로 사용
test_size = len(dataY) - train_size

# 데이터를 잘라 학습용 데이터 생성
trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

# 데이터를 잘라 테스트용 데이터 생성
testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])

# 텐서플로우 플레이스홀더 생성
# 학습용/테스트용으로 X, Y를 생성한다
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
print("X: ", X)
Y = tf.placeholder(tf.float32, [None, 1])
print("Y: ", Y)

# 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
targets = tf.placeholder(tf.float32, [None, 1])
print("targets: ", targets)
predictions = tf.placeholder(tf.float32, [None, 1])
print("predictions: ", predictions)

# 모델(LSTM 네트워크) 생성
def lstm_cell():
    # LSTM셀을 생성한다.
    # num_units: 각 Cell 출력 크기
    # forget_bias: The bias added to forget gates.
    # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
    # state_is_tuple: False ==> they are concatenated along the column axis.
    # cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.sigmoid)
    # cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, forget_bias=0.8, state_is_tuple=True, activation=tf.tanh)
    return cell

# 몇개의 층으로 쌓인 Stacked RNNs 생성, 여기서는 1개층만
multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(1)], state_is_tuple=True)

# RNN Cell(여기서는 LSTM셀임)들을 연결
hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("hypothesis: ", hypothesis)

Y_pred = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_dim, activation_fn=None)


loss = tf.reduce_sum(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE(Root Mean Square Error)
# rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions))) # 아래 코드와 같다
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # 학습한다
    for epoch in range(epoch_num):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print("[step: {} loss: {}".format(epoch, step_loss))

    # 테스트한다
    test_predict = sess.run(Y_pred, feed_dict={X: testX})

    # 테스트용 데이터 기준으로 측정지표 rmse를 산출한다
    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("rmse: ", rmse_val)

    plt.plot(testY, 'r')
    plt.plot(test_predict, 'b')
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()