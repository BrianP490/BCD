import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# import pickle5 as pickle    # integrated into python 3.8+
import pickle 

def create_model(data):

    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train the model
    model = LogisticRegression()
    model.fit(X_train,y_train)

    # test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print('Classification report: \n', classification_report(y_test, y_pred))


    return model, scaler

def get_clean_data():
    """
    Load and clean the dataset from a CSV file.

    This function reads a CSV file containing medical data, performs cleaning operations by
    removing unnecessary columns, and encodes the diagnosis labels into binary format. 

    The specific cleaning operations performed are:
    - Removing the 'Unnamed: 32' and 'id' columns, which are not needed for analysis.
    - Mapping the 'diagnosis' column values from categorical ('M' for malignant and 'B' for benign)
      to binary values (1 for malignant and 0 for benign).

    Returns:
        pd.DataFrame: A cleaned DataFrame containing the relevant medical data with
                      the 'diagnosis' column encoded as binary values.

    Raises:
        FileNotFoundError: If the specified CSV file cannot be found.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV file cannot be parsed.
    """
    # Load the dataset from the specified CSV file
    data = pd.read_csv("data/data.csv")
    
    # Remove unnecessary columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    # Map the diagnosis column values to binary format
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    # Return the cleaned DataFrame
    return data

def main():
    data = get_clean_data()

    model, scaler = create_model(data)


    with open('model/model.pkl', 'wb') as f:
        # Serialization (also known as pickling) is the process of converting a Python object into a byte stream, allowing you to save it to a file or send it over a network
        # f is the file object opened
        pickle.dump(model, f)
        
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

def hello():
    """
    Prints a greeting.
    Used to test if import is working
    """
    print("hello from imports")

# This is checking if the process/thread running this code for an attribute called '__name__'. If the attribute is '__main__' then it executes this code which calls the main function.
# It prevents code from being run during an import to another program/script
# When the script is imported as a module in another script, __name__ is set to the name of the module (i.e., the filename without the .py extension; in this case it will be just 'main').
if __name__ == "__main__":
    main()