import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

labelled_data = pd.read_csv('monthly-data-labelled.csv')
unlabelled_data = pd.read_csv('monthly-data-unlabelled.csv')

    
X = labelled_data.drop(columns=['city', 'year'])
y = labelled_data['city']
    
#normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_unlabelled_scaled = scaler.transform(unlabelled_data.drop(columns=['city', 'year']))

    
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y)
    
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

y_validation = model.predict(X_val)
score = accuracy_score(y_val, y_validation)
print(f"validation score: {score}")
    

predictions = model.predict(X_unlabelled_scaled)

pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)


#df = pd.DataFrame({'truth': y_val, 'prediction': model.predict(X_val)})
#print(df[df['truth'] != df['prediction']])