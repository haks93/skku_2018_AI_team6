clear ; close all;

%% 데이터 읽기
X = csvread('xtrain3.csv');
X = X(:, 2:end);
y = csvread('ytrain.csv');
xtest = csvread('xtest3.csv');
xtest = xtest(:, 2:end);
ytest = csvread('ytest.csv');

%나이, 운임 feature scaling
mu_age = mean(X(:, 2));
sigma_age = std(X(:, 2));
X(:, 2) = (X(:, 2) - mu_age) ./ sigma_age;

mu_fare = mean(X(:, 3));
sigma_fare = std(X(:, 3));
X(:, 3) = (X(:, 3) - mu_fare) ./ sigma_fare;

mu_age2 = mean(xtest(:, 2));
sigma_age2 = std(xtest(:, 2));
xtest(:, 2) = (xtest(:, 2) - mu_age2) ./ sigma_age2;

mu_fare2 = mean(xtest(:, 3));
sigma_fare2 = std(xtest(:, 3));
xtest(:, 3) = (xtest(:, 3) - mu_fare2) ./ sigma_fare2;

% 변수 초기화
theta = zeros(size(X, 2), 1); %가중치 설정
lambda = 0; %정규화 가중치 설정
learning_rate = [0.001; 0.003; 0.01; 0.03; 0.1; 0.3; 1; 3; 9]; %학습률 설정

%%%%%%%%%%%%%%% 직접 만든 로지스틱회귀와 octave 제공 함수 중 선택하여 성능 비교 %%%%%%%%%%%%%
%직접 코딩한 로지스틱 회귀
for i = 1:length(learning_rate)
  theta = gradientDescent(X, y, theta, learning_rate(i), lambda, 1000);
  
  fprintf('Learning rate: %f\n', learning_rate(i));
  p = predict(theta, X);
  fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
  p = predict(theta, xtest);
  fprintf('Test Accuracy: %f\n\n', mean(double(p == ytest)) * 100);
end

% octave에서 제공하는 함수
options = optimset('GradObj', 'on', 'MaxIter', 1000);
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), theta, options);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
% 훈련 정확도 및 예측 정확도 출력
p = predict(theta, X);
fprintf('Octave Optimze\nTrain Accuracy: %f\n', mean(double(p == y)) * 100);
p = predict(theta, xtest);
fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);
