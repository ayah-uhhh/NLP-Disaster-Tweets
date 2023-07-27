import time
import multiprocessing as mp
from nlpRNN import nlp_rnn
from decimal import Decimal

if __name__ == '__main__':
    print("finding optimized parameters...")
    starttime = time.time()

    pool = mp.Pool(mp.cpu_count(), maxtasksperchild=3)

    best_accuracy = 0
    best_params = None
    optimizer = ['rmsprop', 'sgd', 'adam']
    decimal_range = [Decimal(x) / 10 for x in range(11)]

    arguments = []
    for i in range(3):
        for j in range(100, 150, 50):
            for k in decimal_range:
                arguments.append(['cleaned_train_stop.csv', optimizer[i], j, (10, 1), k, False, False, 5, 32])

    results = [pool.apply_async(nlp_rnn, args=(arg,)) for arg in arguments]

    pool.close()
    #pool.join()

    # get the best results from all combinations
    for result in results:
        output = result.get()
        if output[3] > best_accuracy:
            best_accuracy = output[3]
            best_params = output[0]  # array

    if best_params is not None:
        with open('best_RNN_params.txt', 'w') as file:
            file.write(str(best_params))

    total_time = time.time() - starttime
    print("Best error rate: {}".format(best_accuracy))
    print("Best parameters: {}".format(best_params))
    print("Total time: {}".format(total_time))
