class Utility():
    @classmethod
    def grid_scores_to_df(grid_scores):
        """
        Convert a sklearn.grid_search.GridSearchCV.grid_scores_ attribute to a tidy
        pandas DataFrame where each row is a hyperparameter-fold combinatination.
        """
        rows = list()
        for grid_score in grid_scores:
            for fold, score in enumerate(grid_score.cv_validation_scores):
                row = grid_score.parameters.copy()
                row['fold'] = fold
                row['score'] = score
                rows.append(row)
        df = pd.DataFrame(rows)
        return df