import statistics

def compute_mean_var(a):
    length_1 = len(a[0])
    mean = [0 for i in range(length_1)]
    var = [0 for i in range(length_1)]
    length_0 = len(a)
    for i in range(length_1):
        tmp = []
        for j in range(length_0):
            tmp.append(a[j][i])
        var[i]=(statistics.variance(tmp))
        mean[i]=(sum(tmp)/length_0)
    return mean,var


if __name__ == "__main__":
    a = [
        [0,1,2,3],
        [0,1,2,3],
        [0,1,2,3]
    ]
    mean,var = compute_mean_var(a)
    print(mean,var)

