import os
import io
import base64
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend a plotokhoz
import matplotlib.pyplot as plt

app = Flask(__name__)

# --- FELTÖLTÉSI FOLYAMAT ÉS MODELL LÉTREHOZÁS ---

def create_sample_data():
    """Minta adatok létrehozása demo célból."""
    np.random.seed(42)
    n_samples = 500
    data = {
        'tenure': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Wire transfer', 'Mailed check'], n_samples),
        'total_charges': np.random.uniform(50, 8000, n_samples)
    }
    df = pd.DataFrame(data)
    
    churn_prob = 0.3
    df['churn'] = (np.random.rand(n_samples) < churn_prob).astype(int)
    df.loc[(df['tenure'] < 12) & (df['contract_type'] == 'Month-to-month'), 'churn'] = 1
    
    return df

def load_or_create_model(df, churn_column='churn'):
    """Betölt vagy létrehoz egy modellt az adatok alapján (dinamikus churn oszlopnévvel)."""
    if df is None or df.empty:
        df = create_sample_data()
        churn_column = 'churn'
    
    # Kategorikus oszlopok kódolása (szöveges mezők átalakítása számokká)
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != churn_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Modell betanítása
    X = df.drop(churn_column, axis=1)
    y = df[churn_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, label_encoders, report, conf_matrix

# --- HTML SABLON A WEBES INTERFÉSZHEZ ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="hu">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ügyfél Kiszállás Modell</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --light-gray: #ecf0f1;
            --dark-gray: #34495e;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f5f7fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 20px 0;
            text-align: center;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        h1, h2 {
            text-align: center;
            color: var(--primary-color);
        }
        .content-section {
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 15px;
            padding: 15px;
            background-color: var(--light-gray);
            border-radius: 5px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: var(--dark-gray);
        }
        input[type="file"], input[type="text"], select, button {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
            box-sizing: border-box;
        }
        button {
            background-color: var(--secondary-color);
            color: white;
            cursor: pointer;
            transition: background 0.3s;
            margin-top: 10px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .results-section {
            display: none;
        }
        .chart-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        .chart-box {
            flex: 1;
            min-width: 300px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: var(--primary-color);
            color: white;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        .churn-risk {
            color: var(--accent-color);
            font-weight: bold;
        }
        .churn-safe {
            color: green;
            font-weight: bold;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-box {
            background: #e8f6f3;
            padding: 15px;
            text-align: center;
            border-radius: 5px;
            border: 1px solid #d0e1de;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--primary-color);
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>Ügyfél Kiszállás (Churn) Modell</h1>
            <p>Ez az alkalmazás segít az ügyfélgazdagság előrejelzésében és elemzésében.</p>
        </div>
    </header>

    <div class="container">
        <!-- Feltöltési Szakasz -->
        <section class="content-section">
            <h2>Adatfeltöltés és Modell Betanítás</h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="dataFile">Adatfájl (CSV vagy Excel) feltöltése:</label>
                    <input type="file" id="dataFile" name="file" accept=".csv,.xlsx,.xls">
                </div>
                <div class="form-group">
                    <label for="churnColumn">Kiszállás (churn) oszlop neve:</label>
                    <input type="text" id="churnColumn" name="churn_column" value="Exited" placeholder="Például: Exited vagy churn">
                    <small>Ha nincs ilyen oszlop, a rendszer generál egy mintát.</small>
                </div>
                <button type="button" onclick="processData()">Modell Betanítása és Elemzés</button>
            </form>
        </section>

        <!-- Eredmények Szakasz -->
        <section id="resultsSection" class="content-section results-section">
            <h2>Elemzési Eredmények</h2>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value" id="totalCustomers">--</div>
                    <div>Összes Ügyfél</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="churnCount">--</div>
                    <div>Kiszálló Ügyfelek</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="churnRate">--</div>
                    <div>Kiszállási Ráta (%)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="modelAccuracy">--</div>
                    <div>Modell Pontosság (%)</div>
                </div>
            </div>

            <div class="chart-container">
                <div class="chart-box">
                    <h3>Kiszállási Ráta Megoszlása</h3>
                    <div id="churnDistributionChart"></div>
                </div>
                <div class="chart-box">
                    <h3>Modell Pontossága (Confusion Matrix)</h3>
                    <div id="confusionMatrixChart"></div>
                </div>
            </div>

            <h3>Előrejelzett Kiszálló Ügyfelek Részletes Listája</h3>
            <div id="churnedCustomersTable"></div>
        </section>
    </div>

    <script>
        async function processData() {
            const fileInput = document.getElementById('dataFile');
            const churnColumnInput = document.getElementById('churnColumn');
            const file = fileInput.files[0];
            const churnColumn = churnColumnInput.value.trim() || 'churn';

            if (!file) {
                alert('Kérem, válasszon egy fájlt.');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);
            formData.append('churn_column', churnColumn);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || `Hiba történt: ${response.statusText}`);
                }

                displayResults(data);

            } catch (error) {
                console.error('Hiba:', error);
                alert('Hiba történt a feldolgozás közben: ' + error.message);
            }
        }

        function displayResults(data) {
            document.getElementById('resultsSection').style.display = 'block';

            // Statisztikák
            document.getElementById('totalCustomers').textContent = data.total_customers;
            document.getElementById('churnCount').textContent = data.churn_count;
            document.getElementById('churnRate').textContent = data.churn_rate.toFixed(2);
            document.getElementById('modelAccuracy').textContent = data.model_accuracy.toFixed(2);

            // Kiszállási eloszlás diagram (Plotly)
            const churnTrace = {
                labels: ['Nem kiszálló', 'Kiszálló'],
                values: [data.non_churn_count, data.churn_count],
                type: 'pie',
                marker: {
                    colors: ['green', '#e74c3c']
                }
            };
            const churnLayout = {
                title: 'Kiszállási Ráta Megoszlása',
                height: 400
            };
            Plotly.newPlot('churnDistributionChart', [churnTrace], churnLayout);

            // Confusion Matrix diagram (Plotly Heatmap)
            const confMatrixTrace = {
                z: data.confusion_matrix,
                x: ['Előrejelzett Nem', 'Előrejelzett Igen'],
                y: ['Valódi Nem', 'Valódi Igen'],
                type: 'heatmap',
                colorscale: 'Blues'
            };
            const confMatrixLayout = {
                title: 'Konfúziós Mátrix',
                xaxis: { title: 'Előrejelzett Érték' },
                yaxis: { title: 'Valódi Érték' }
            };
            Plotly.newPlot('confusionMatrixChart', [confMatrixTrace], confMatrixLayout);

            // Ügyféllista
            const churnedCustomersTableBody = document.getElementById('churnedCustomersTable');
            
            if (!data.customers || data.customers.length === 0) {
                churnedCustomersTableBody.innerHTML = '<p>Nincs előrejelzett kiszálló ügyfél.</p>';
                return;
            }

            let tableHTML = '<table><thead><tr>';
            
            // Oszlopfejlécek (Spread operátorral javítva)
            const columns = ['Index', ...Object.keys(data.customers[0])];
            columns.forEach(col => {
                tableHTML += `<th>${col}</th>`;
            });
            tableHTML += '</tr></thead><tbody>';

            // Sorok
            data.customers.forEach(customer => {
                tableHTML += '<tr>';
                const values = Object.values(customer);
                tableHTML += `<td>${values.shift()}</td>`; // Index
                values.forEach(val => {
                    if (typeof val === 'boolean') {
                        const style = val ? 'churn-risk' : 'churn-safe';
                        tableHTML += `<td class="${style}">${val ? 'Igen' : 'Nem'}</td>`;
                    } else {
                        tableHTML += `<td>${val}</td>`;
                    }
                });
                tableHTML += '</tr>';
            });

            tableHTML += '</tbody></table>';
            churnedCustomersTableBody.innerHTML = tableHTML;
        }
    </script>
</body>
</html>
"""

# --- PYTHON BACKEND LOGIKA ---

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict_churn():
    try:
        file = request.files.get('file')
        churn_column = request.form.get('churn_column', 'Exited').strip() if request.form.get('churn_column') else 'Exited'

        if not file:
            return jsonify({"error": "Nincs fájl kiválasztva."}), 400

        # Adatbetöltés pandas DataFrame-be
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                return jsonify({"error": "Csak CSV vagy Excel fájlokat támogatunk."}), 400
        except Exception as e:
            return jsonify({"error": f"Hiba történt a fájl beolvasásakor: {str(e)}"}), 400

        # Modell betanítása / betöltése a megadott churn oszlopnévvel
        if churn_column in df.columns:
            model, label_encoders, report, conf_matrix = load_or_create_model(df.copy(), churn_column)
        else:
            df = create_sample_data()
            churn_column = 'churn'
            model, label_encoders, report, conf_matrix = load_or_create_model(df.copy(), churn_column)

        # Adatok előkészítése az előrejelzéshez
        X = df.drop(churn_column, axis=1, errors='ignore')
        y_true = df[churn_column] if churn_column in df.columns else None
        
        # Kategorikus oszlopok átalakítása az előrejelzéshez
        for col in X.select_dtypes(include='object').columns:
            if col in label_encoders:
                X[col] = label_encoders[col].transform(X[col].astype(str))
            else:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # Előrejelzés
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        # Eredmények összeállítása
        results_df = X.copy()
        results_df['Actual_Churn'] = y_true if y_true is not None else 'N/A'
        results_df['Predicted_Churn'] = predictions
        results_df['Churn_Probability'] = probabilities

        # Statisztikák
        churn_count = int(results_df['Predicted_Churn'].sum())
        total_customers = len(results_df)
        churn_rate = (churn_count / total_customers) * 100
        accuracy = (report['accuracy'] * 100) if isinstance(report, dict) else 0

        # Kiszálló ügyfelek listája
        churned_customers_df = results_df[results_df['Predicted_Churn'] == 1].head(20)
        churned_customers_list = []
        for idx, (_, row) in enumerate(churned_customers_df.iterrows()):
            customer_dict = row.to_dict()
            customer_dict['Index'] = idx + 1
            churned_customers_list.append(customer_dict)

        conf_matrix_list = conf_matrix.tolist()

        return jsonify({
            "total_customers": total_customers,
            "churn_count": churn_count,
            "non_churn_count": total_customers - churn_count,
            "churn_rate": churn_rate,
            "model_accuracy": accuracy,
            "confusion_matrix": conf_matrix_list,
            "customers": churned_customers_list
        })

    except Exception as e:
        return jsonify({"error": f"Váratlan hiba történt: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)