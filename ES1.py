def ES1(dataframe):
    subset = []
    not_subset = []
    q_2 = np.array(dataframe["q2"])
    for i in range(len(q_2)):
        if (8 < q_2[i]) and (q_2[i] < 11):
            not_subset.append(i)
            continue
        elif (12.5 < q_2[i]) and (q_2[i] < 15):
            not_subset.append(i)
            continue
        else:
            subset.append(i)

    subset = dataframe.iloc[subset]

    not_subset = dataframe.iloc[not_subset]

    return subset, not_subset
