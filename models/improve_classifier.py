import argparse
import sys
import numpy as np
from pprint import pprint
from sklearn.model_selection import GridSearchCV, train_test_split
from models.train_classifier import build_model, load_data, evaluate_model, save_model, build_pipeline_for_category


def build_grid(parameters=None, show_progress=True, cv=2):
    """
    Create GridSearchCV object based on pipeline from train_classifier
    :param parameters: optional dict with parameters and ranges for grid search
    :param show_progress: displays verbosity messages during grid search
    :param cv: number of cross valalidation folds - defaulted to minimum of 2
    :return: GridSearchCV instance
    """
    if not parameters:
        parameters = {
            'clf__estimator__n_estimators': [10],
            'clf__estimator__max_depth': [500]
        }

    verbose = 5 if show_progress else 0
    pipeline = build_model()

    grid = GridSearchCV(pipeline, param_grid=parameters, cv=cv, verbose=verbose)
    return grid


def get_parser():
    """
    Parser that defines command line parameters
    Supports two commands:
        show_params - print all possible parameters for Grid Search object
        grid_search - perform Grid search using set of the parameters provided or using default if omitted
    :return: argparser instance
    """
    parser = argparse.ArgumentParser(description='Train multiple models using Grid Search',
                                     epilog='Example: python ' \
                                            'improve_classifier.py ../data/DisasterResponse.db classifier.pkl')
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('show_params')

    cmd_train = subparsers.add_parser('grid_search')
    cmd_train.add_argument('database_filepath', help='Filepath of the disaster messages database')
    cmd_train.add_argument('model_filepath', help='Pickle filename for storing best model')
    cmd_train.add_argument('-p', required=False, dest='parameters',
                           help='Sets parameters and ranges for grid search in form of dict. If not set the default is used.')
    cmd_train.add_argument('-v', action='store_true', help='Verbose output', dest='verbose')

    cmd_category = subparsers.add_parser('evaluate_category')
    cmd_category.add_argument('database_filepath', help='Filepath of the disaster messages database')
    cmd_category.add_argument('category_name')

    return parser



def calculate_calss_weights(Y_train):
    class_weights =[]
    for idx in range(Y_train.shape[1]):
        counts = np.bincount(Y_train[:, idx])
        classes = np.arange(len(counts))
        class_weights.append({cl[0]:np.sum(counts)/cl[1] for cl in np.column_stack((classes, counts))})

    return class_weights



def train_evaluate_one_category(X_train, Y_train, X_test, Y_test, class_weight, ctagory_name):
    """ Train and evaluate performnce of the model using class_weight

    :param X_train: Free text for training
    :param Y_train: Category labels for training
    :param X_test: Free text for testing
    :param Y_test: Category labels for testing
    :param class_weight: dict with weight ration per predicted class
    :param ctagory_name: category name from database for display purpose
    :return: None
    """
    model = build_pipeline_for_category(class_weight)
    print('Built category model')
    model.fit(X_train, Y_train)
    print('Fit category model')
    evaluate_model(model, X_test, Y_test, [ctagory_name])


def main():
    parser = get_parser()
    # args = parser.parse_args(['show_params'])
    # args = parser.parse_args(['evaluate_category','../data/DisasterResponse.db', 'offer'])
    # args = parser.parse_args(['grid_search', '../data/DisasterResponse.db', 'classifier.pkl','-p', '{"a":[1]}'])
    # args = parser.parse_args(['grid_search', '../data/DisasterResponse.db', 'classifier.pkl', '-v'])

    args = parser.parse_args()

    if args.command == 'show_params':
        grid = build_grid()
        pprint(grid.get_params())
        return

    print('Loading data...\n    DATABASE: {}'.format(args.database_filepath))
    X, Y, category_names = load_data(args.database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    if args.command == 'evaluate_category':
        # doing single category
        cat_idx = np.argwhere(np.array(category_names) == args.category_name)
        if not cat_idx:
            print(f'Cannot find category:{args.category_name}')
            sys.exit(-1)
        cat_idx = cat_idx[0][0]
        class_weight = calculate_calss_weights(Y_train)
        train_evaluate_one_category(X_train, Y_train[:, cat_idx], X_test, Y_test[:, cat_idx],
                                    class_weight[cat_idx],
                                    category_names[cat_idx])
        return

    print('Building grid...')
    parameters = eval(args.parameters) if args.parameters else None
    model = build_grid(parameters=parameters, show_progress=args.verbose)

    print('Training multiple models and choosing the best one...')
    model.fit(X_train, Y_train)

    print('Best_model_parametsrs')
    print(model.best_params_)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(args.model_filepath))
    save_model(model, args.model_filepath)

    print('Best model saved!')


if __name__ == '__main__':
    main()