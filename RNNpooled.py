#pooled RNN
import time
import multiprocessing as mp
from nlpRNN import nlp_rnn
from decimal import *
if __name__ == '__main__':
    
    print("finding optimized parameters...")
    
    starttime = time.time()
    
    pool = mp.Pool(mp.cpu_count(), maxtasksperchild = 3)
    
    best_accuracy = 0
    optimizer = ['rmsprop', 'sgd', 'adam']
    decimal_range = [Decimal(x) / 10 for x in range(11)]
    for i in range(3):
        for j in range(100,500, 50):
            #for k in range(0.0,1.0,0.1):
            for k in decimal_range:
                results = [pool.apply_async(
                    nlp_rnn, args=(['cleaned_train_stop.csv', optimizer[i], 256, (10,1), k, False, False, j, 64]))]

    pool.close()

    # get the results after each thread has finished executing
    output = []
    for job in results:
        output.append(job.get())

    for x in output:
        if (x[3] > best_accuracy):
            best_accuracy = x[3]
            best_params = x[0]  # array
            best_optimizer = x[0][0]
            best_units = x[0][1]
            best_input_shape = x[0][2]
            best_epochs = x[0][3]
            best_dropout_rate = x[0][4]
            model = x[4]

        if best_params is not None:
            with open('best_RNN_params.txt', 'w') as file:
                file.write(str(best_params))
    total_time = time.time() - starttime
    print("Best error rate: {}".format(best_accuracy))
    print("Best optimizer: {}".format(best_optimizer))
    print("Total time: {}".format(total_time))