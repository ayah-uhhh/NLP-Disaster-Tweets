#100 tests
import multiprocessing as mp
import time
from nlpRNN import nlp_rnn
import matplotlib as plt

if __name__ == '__main__':
    print("Running 100 iterations of the RNN model...")
    starttime = time.time()
    pool = mp.Pool(mp.cpu_count(), maxtasksperchild=3)
    results = []

    for _ in range(100):
        result = pool.apply_async(
            nlp_rnn, args=('adam', 128, (10, 1), False, False, 200, 32))
        results.append(result)

    pool.close()
    pool.join()

    total_accuracy = 0
    accuracy_values = []

    for result in results:
        accuracy = result.get()
        total_accuracy += accuracy
        accuracy_values.append(accuracy)

    average_accuracy = total_accuracy / len(results)
    total_time = time.time() - starttime
    print("Average accuracy: {}".format(average_accuracy))
    print("Total time: {}".format(total_time))
    
    plt.plot(accuracy_values)
    plt.title('Average Accuracy across 100 iterations: Cleaned and Stopwords Included')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.show()
