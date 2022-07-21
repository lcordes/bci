# def loocv_plot(users, ax, config):
#     data = np.zeros((len(classifiers), len(users)))
#     classes = config["n_classes"]
#     conf_matrices = [np.zeros((classes, classes)) for _ in range(3)]

#     for row, constructor in enumerate(constructors):
#         for column, recording_name in enumerate(users):
#             metrics = score_loocv(recording_name, constructor, config)
#             data[row, column] = metrics["acc"]
#             conf_matrices[row] = conf_matrices[row] + metrics["conf"]
#             print(f"Classifier {row+1} user {column+1} done")

#     # Add row and column means
#     data = np.concatenate((data, np.mean(data, axis=0).reshape(1, -1)), axis=0)
#     data = np.concatenate((data, np.mean(data, axis=1).reshape(-1, 1)), axis=1)

#     # Show the tick labels
#     ax.xaxis.set_tick_params(labeltop=True)

#     x_labels = users + ["avg"]
#     y_labels = [(c.__name__).replace("Classifier", "") for c in constructors] + ["avg"]
#     # Hide the tick labels
#     ax.xaxis.set_tick_params(labelbottom=False)
#     sns.heatmap(
#         data,
#         ax=ax,
#         vmin=0,
#         vmax=1,
#         annot=True,
#         xticklabels=x_labels,
#         yticklabels=y_labels,
#     )


# def save_loocv_plot(users, config, title):
#     fig, _ = plt.subplots(1, 1, figsize=(15, 3))
#     axes = fig.axes

#     loocv_plot(users, axes[0], config)
#     fig.tight_layout()
#     plt.title(title)
#     plt.savefig(
#         f"{RESULTS_PATH}/loocv_within_users/{title}.png", dpi=400, bbox_inches="tight"
#     )


    # Create permutations separately per classifier to account for clf specific parameters
    for clf in clfs:
        if clf in clf_specific:
            all_params = {**hyper_grid, **clf_specific[clf], "model_type": clf}
        else:
            all_params = {**hyper_grid}
        permutations.extend(list(ParameterGrid(all_params)))
