from matplotlib import pyplot as plt


def count_string_occurrences(strings, target_string):
    return strings.count(target_string)


def explain(decisionTreeTestResults):
    conditions = decisionTreeTestResults[0][1]
    total_attributes = []
    for i in conditions:
        attribute, value = i.split(" <= ")
        total_attributes.append(attribute)

    no_of_times = {}
    for i in total_attributes:
        target_string = i
        occurrences = count_string_occurrences(total_attributes, target_string)
        no_of_times[i] = occurrences
    cum_sum = sum(no_of_times.values())

    percentage = {}
    for i, j in zip(no_of_times.keys(), no_of_times.values()):
        target_string = i
        percentage[i] = j / cum_sum

    keys = list(percentage.keys())
    values = list(percentage.values())
    plt.barh(keys, values)
    plt.show()
    plt.savefig('figure.png')

    sorted_des_dict = dict(sorted(percentage.items(), key=lambda x: x[1], reverse=True))

    print("Explanation of the prediction got : ")
    for i, j in zip(percentage.keys(), range(1, len(sorted_des_dict) + 1)):
        if j == len(sorted_des_dict):
            print(
                "'{}' feature has {} highest impact for prediction got that is '{}' and rest all features have no impact on the prediction got".format(
                    i, j, decisionTreeTestResults[0][0]), end='')
            continue
        print("'{}' feature has {} highest impact ,".format(i, j), end='')

