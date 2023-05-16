import numpy as np

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    ###################
    spam = matrix[category == 1, :]
    nonspam = matrix[category == 0, :]

    phi_y1 = (np.sum(spam, axis = 0) + 1) / (spam.sum() + N)
    phi_y0 = (np.sum(nonspam, axis = 0) + 1) / (nonspam.sum() + N)
    phi_y = spam.shape[0] / matrix.shape[0]

    state['phi_y1'] = phi_y1    # prob that jth token appears given y=1
    state['phi_y0'] = phi_y0    # prob that jth token appears given y=1
    state['phi_y'] = phi_y

    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################

    phi_y1 = state['phi_y1']
    phi_y0 = state['phi_y0']
    phi_y = state['phi_y']

    log_probSpam = np.sum(np.log(phi_y1)*matrix, axis = 1) + np.log(phi_y)
    log_probNon = np.sum(np.log(phi_y0)*matrix, axis = 1) + np.log(phi_y)

    output[log_probSpam >= log_probNon] = 1
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('data\MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('data\MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()
