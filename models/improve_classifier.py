import argparse
from pprint import pprint
from sklearn.model_selection import GridSearchCV, train_test_split
from models.train_classifier import build_model, load_data, evaluate_model, save_model


def build_grid(parameters=None, show_progress=True, cv=2):
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
    return parser


def main():
    parser = get_parser()

    #args = parser.parse_args(['show_params'])
    #args = parser.parse_args(['grid_search', '../data/DisasterResponse.db', 'classifier.pkl','-p', '{"a":[1]}'])
    args = parser.parse_args(['grid_search', '../data/DisasterResponse.db', 'classifier.pkl', '-v'])

    if args.command == 'show_params':
        grid = build_grid()
        pprint(grid.get_params())
        return

    print('Loading data...\n    DATABASE: {}'.format(args.database_filepath))
    X, Y, category_names = load_data(args.database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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