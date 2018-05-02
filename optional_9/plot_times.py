import numpy as np
import matplotlib.pyplot as plt

with open('timing_data/train_time_GP.csv') as f:
    lines = f.readlines()
    GP_train = np.array(lines, dtype = np.float32)


with open('timing_data/test_time_GP.csv') as f:
    lines = f.readlines()
    GP_test = np.array(lines, dtype = np.float32)


with open('timing_data/train_time_Ridge.csv') as f:
    lines = f.readlines()
    Ridge_train = np.array(lines, dtype = np.float32)


with open('timing_data/test_time_Ridge.csv') as f:
    lines = f.readlines()
    Ridge_test = np.array(lines, dtype = np.float32)

print(len(GP_train))
print(len(GP_test))
print(len(Ridge_train))
print(len(Ridge_test))


plt.hist(GP_train)
plt.xlabel('Training Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of GPR Training')
plt.savefig('graphs/GP_train.png')
plt.show()


plt.hist(GP_test)
plt.xlabel('Testing Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of GPR Testing')
plt.savefig('graphs/GP_test.png')
plt.show()


plt.hist(Ridge_train)
plt.xlabel('Training Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of Ridge Regression Training')
plt.savefig('graphs/Ridge_train.png')
plt.show()


plt.hist(Ridge_test)
plt.xlabel('Testing Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of Ridge Regression Testing')
plt.savefig('graphs/Ridge_test.png')
plt.show()






