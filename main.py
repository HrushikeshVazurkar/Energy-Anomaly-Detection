import threading
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import queue
from collections import deque
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import beta
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import warnings

warnings.filterwarnings('ignore')

stop_event = threading.Event() # THREAD STOPPING EVENT
start_viz = threading.Event() # VIZ STARTING EVENT

# CLASS DATA_GENERATION
# This class is used to generate the data synthetically for the dynamic prices of energy on hourly basis.
# SUBCLASS ANOMALY - Used to add anomalous instances to the data.
# SUBCLASS TIME_SERIES - Used to add time-series properties like trend/drift and seasonality to the synthetic data.
# SUBCLASS BASE - Used to generate the synthetic data, which is used by downstream tasks.
class DATA_GENERATION:
    class ANOMALY:

        # This method simulates instances of global anomalous behaviour disrupting the energy prices - sudden drop/rise due to geopolitical events etc.
        @staticmethod
        def apply_global_anomaly(price):
            anomaly_type = random.choice(['global_spike', 'global_dip', 'none'])
            
            if anomaly_type == 'global_spike':
                anomaly_factor = random.uniform(1.5, 3)  # Apply a global price spike (e.g., a geopolitical event)
                price *= anomaly_factor
                print(f"Global price spike! New price: €{price:.2f}/MWh")
            
            elif anomaly_type == 'global_dip':
                anomaly_factor = random.uniform(0.3, 0.5)  # Apply a global price dip (e.g., oversupply of energy)
                price *= anomaly_factor
                print(f"Global price dip! New price: €{price:.2f}/MWh")
            
            return price
            
        # This method applies contextual anomalies - subtle anomalies depending on a short time-frame due to various local factors.
        @staticmethod
        def apply_contextual_anomaly(price, hour, anomaly_type, PEAK_HOURS):
                PEAK_HOURS = range(7, 22)  # Peak hours - 7 AM to 9 PM

                # Contextual anomaly in peak hours
                if anomaly_type == 'context_spike' and hour in PEAK_HOURS:
                    anomaly_factor = random.uniform(1.2, 1.5)
                    price *= anomaly_factor

                # Contextual anomaly in off-peak hours
                elif anomaly_type == 'context_negative' and hour not in PEAK_HOURS:
                    anomaly_factor = random.uniform(0.6, 0.8)
                    price *= anomaly_factor
                
                return price

    class TIME_SERIES:
        DAILY_SEASONALITY_AMPLITUDE = 5  # Daily price oscillation range
        YEARLY_SEASONALITY_AMPLITUDE = 10  # Yearly price oscillation range
        YEARLY_CYCLE_PERIOD = 365  # Days in a year

        # This method adds a component of seasonality to the data based on the hour of energy price calculation.
        @staticmethod
        def add_daily_seasonality(hour):
            # Beta Distribution Parameters
            ALPHA = 2  # Shape parameter (controls the peak)
            BETA = 3  # Shape parameter (controls the spread)
            HOUR_SCALE = 24

            normalized_hour = hour / HOUR_SCALE
            daily_seasonality = beta.pdf(normalized_hour, ALPHA, BETA)
            return daily_seasonality * 10  # Scale to a range suitable for your data

        # This method adds a component of seasonality to the data based on the day of week of energy price calculation.
        @staticmethod
        def add_weekly_seasonality(day_of_week):
            ALPHA = 2  # Shape parameter (controls the peak)
            BETA = 3   # Shape parameter (controls the spread)
            DAY_SCALE = 7

            normalized_day = day_of_week / DAY_SCALE
            weekly_seasonality = beta.pdf(normalized_day, ALPHA, BETA)
            return weekly_seasonality * 10  # Scale to a range suitable for your data

        # This method adds drift/trend to the data based on month of year - higher prices in winter etc.
        @staticmethod
        def add_trend(month):
            TREND_PEAK_MONTHS = [12, 1, 2]  # December, January, February
            MEAN_MONTH = 1.0  # Center of the Gaussian peak (February)
            STD_DEV_MONTH = 2.0  # Standard deviation to spread the trend (approximately 2 months)

            month_float = float(month)
            # Calculate Gaussian distribution centered around February (month 2) with a standard deviation
            seasonal_trend = (1 / (STD_DEV_MONTH * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((month_float - MEAN_MONTH) ** 2) / (STD_DEV_MONTH ** 2))
            return seasonal_trend * 100  # Scale the trend to appropriate range

    class BASE:
    
        # This method calculates energy prices with trend/drift + seasonality
        @staticmethod
        def get_energy_price(timestamp):
            PEAK_MEAN_PRICE = 60; OFFPEAK_MEAN_PRICE = 40 # Peak and Off-peak price in Euros
            PEAK_STD_DEV = 10; OFFPEAK_STD_DEV = 5  # Peak and Off-peak Standard deviation for energy prices
            WINTER_ADJUSTMENT = 1.3  # 30% increase in winter
            WEEKEND_DISCOUNT = 0.9  # 10% reduction on weekends
            PEAK_HOURS = range(7, 22)  # Peak hours - 7 AM to 9 PM

            hour = timestamp.hour
            day_of_week = timestamp.weekday()  # 0 = Monday, 6 = Sunday
            month = timestamp.month
            
            if month in [12, 1, 2]: # Winter Adjustment
                peak_price = PEAK_MEAN_PRICE * WINTER_ADJUSTMENT
                offpeak_price = OFFPEAK_MEAN_PRICE * WINTER_ADJUSTMENT
            else:
                peak_price = PEAK_MEAN_PRICE
                offpeak_price = OFFPEAK_MEAN_PRICE
            
            if day_of_week in [5, 6]:
                peak_price *= WEEKEND_DISCOUNT
                offpeak_price *= WEEKEND_DISCOUNT
            
            if hour in PEAK_HOURS: # Peak vs Off-peak
                price = np.random.normal(peak_price, PEAK_STD_DEV)
            else:
                price = np.random.normal(offpeak_price, OFFPEAK_STD_DEV)
            
            trend = DATA_GENERATION.TIME_SERIES.add_trend(month)
            daily_seasonality = DATA_GENERATION.TIME_SERIES.add_daily_seasonality(hour)
            weekly_seasonality = DATA_GENERATION.TIME_SERIES.add_weekly_seasonality(day_of_week)
            final_price = price + trend + daily_seasonality + weekly_seasonality
            
            return max(final_price, 10)  # Ensure the price is non-negative

        # This method simulates anomalous real-time behaviour for some energy price instances.
        @staticmethod
        def get_energy_price_with_anomalies(timestamp):
            price = DATA_GENERATION.BASE.get_energy_price(timestamp); hour = timestamp.hour
            PEAK_HOURS = range(7, 22)  # Peak hours - 7 AM to 9 PM

            global_anomaly = False; contextual_anomaly = False
            if random.random() < 0.01:  # Randomly decide if a global anomaly occurs. 1% chance of a global anomaly
                price = DATA_GENERATION.ANOMALY.apply_global_anomaly(price)
                global_anomaly = True
            
            anomaly_type = random.choices(['context_spike', 'context_negative', 'none'], weights=[0.05, 0.05, 0.9], k=1)[0]
            if anomaly_type != 'none':
                price = DATA_GENERATION.ANOMALY.apply_contextual_anomaly(price, hour, anomaly_type, PEAK_HOURS)  # Apply contextual anomalies based on specific conditions
                contextual_anomaly = True

            return price, global_anomaly or contextual_anomaly

        # This method is a generator function for the energy prices, starting from 1 Jan, 2024.
        @staticmethod
        def generate_energy_prices():
            
            current_date = datetime(2024, 1, 1, 1, 0)
            while not stop_event.is_set():
                base_price, anomaly = DATA_GENERATION.BASE.get_energy_price_with_anomalies(current_date)
                yield {
                    'timestamp': current_date,
                    'date': current_date.date(),
                    'hour': current_date.hour,
                    'day': current_date.weekday(),
                    'value': base_price,
                    'anomaly': anomaly
                }
                current_date += timedelta(hours=1)
                time.sleep(0.01)  # Sleep to control the speed of number generation

class ML:

    # This method trains and tests the ML model - Isolation Forest for anomaly detection.
    @staticmethod
    def algo():
        print("Training starts")
        labels = [item['anomaly'] for item in eval_q]

        df = pd.DataFrame({
            'timestamp': [item['timestamp'] for item in eval_q],
            # 'date': [item['date'] for item in eval_q],
            'day': [item['day'] for item in eval_q],
            'hour': [item['hour'] for item in eval_q],
            'value': [item['value'] for item in eval_q]
        })
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size=0.2, random_state=42)

        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        model.fit(train_df)

        predicted_anomalies = [True if x == -1 else False for x in model.predict(test_df)]

        print(predicted_anomalies)
        report = classification_report(test_labels, predicted_anomalies, target_names=['Normal', 'Anomaly'])
        print(report)

        with open('isolation_forest_model.pkl', 'wb') as file:
            pickle.dump(model, file)

    # This method returns model inferences - anomaly or not.
    @staticmethod
    def inference(data_point):
        try:
            with open('isolation_forest_model.pkl', 'rb') as file:
                model = pickle.load(file)
                new_data = np.array([[data_point['day'], data_point['hour'], data_point['value']]])
                return model.predict(new_data)[0] == -1
        except Exception as e:
            print("Trained model not found. Please train the model before inference.")
            return False
        
class THREAD_FUNCTIONS:

    # This method runs independently in a thread, generating energy prices.
    @staticmethod
    def data_generator(q):
        for data_point in DATA_GENERATION.BASE.generate_energy_prices():
            q.put(data_point)
        q.put(None)     # Put a sentinel value to signal completion

    # This method consumes the energy prices generated and trains/produces inferences from the Isolation Forest model,
    # and gives a prediction whether anomalous or not.
    @staticmethod
    def data_consumer(q, viz_q):
        global data_collection
        global start_viz
        print("Historical Data Collection - 1 year.")
        while not stop_event.is_set():
            data_point = q.get()
            if len(eval_q) < 100 and data_collection == True:
                eval_q.append(data_point)
            elif len(eval_q) == 100 and data_collection == True:
                data_collection = False
                print("\nModel Training:")
                ML.algo()
                eval_q.clear()
            elif data_collection == False:
                data_point['model_anomaly'] = ML.inference(data_point)
                eval_q.append(data_point); viz_q.append(data_point)
                start_viz.set()
                                        
            time.sleep(0.1)
        
    # This method plots the visualisation for latest 50 energy price points, highlighting anomalous instances(red).
    @staticmethod
    def data_viz():
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'r-')
        
        ax.set_ylim(0, 200)
        ax.set_xlim(0, 50)

        highlighted_points, = ax.plot([], [], 'ro', label='First Hour of Day')
        anomaly_points, = ax.plot([], [], 'go', label='Anomalies')

        while not stop_event.is_set():
            try:
                line.set_xdata(range(len(viz_q)))
                line.set_ydata([i['value'] for i in list(viz_q)])

                for annotation in reversed(ax.texts):
                    annotation.remove()

                # Highlight start of each day (hour == 0)
                first_hour_data = [(idx, i['value'], i['timestamp']) for idx, i in enumerate(viz_q) if i['timestamp'].hour == 0]
                if first_hour_data:
                    first_hour_x = [x[0] for x in first_hour_data]
                    first_hour_y = [x[1] for x in first_hour_data]
                    first_hour_dates = [x[2] for x in first_hour_data]

                    highlighted_points.set_xdata(first_hour_x)
                    highlighted_points.set_ydata(first_hour_y)

                    for x, y, date in zip(first_hour_x, first_hour_y, first_hour_dates):
                        ax.text(x, y, date.strftime('%Y-%m-%d %H'), fontsize=8, color='black', ha='right')

                # Highlight anomalies
                anomaly_data = [(idx, i['value']) for idx, i in enumerate(viz_q) if i.get('model_anomaly', True)]
                if anomaly_data:
                    anomaly_x = [x[0] for x in anomaly_data]
                    anomaly_y = [x[1] for x in anomaly_data]

                    anomaly_points.set_xdata(anomaly_x)
                    anomaly_points.set_ydata(anomaly_y)

                ax.set_xlim(0, len(viz_q))
                fig.canvas.draw()
                fig.canvas.flush_events()

                plt.pause(0.01)
            except:
                continue

            time.sleep(1)

        plt.ioff()
        plt.show()

q = queue.Queue(); viz_q = deque(maxlen=50); eval_q = deque(maxlen=8760)
data_collection = True
lock = threading.Lock()

# Producer + consumer thread start
generator_thread = threading.Thread(target=THREAD_FUNCTIONS.data_generator, args=(q,)); generator_thread.start()
consumer_thread = threading.Thread(target=THREAD_FUNCTIONS.data_consumer, args=(q, viz_q)); consumer_thread.start()

# Main thread execution - Data Viz
try:
    while True:
        if start_viz.is_set():
            THREAD_FUNCTIONS.data_viz()
            time.sleep(1)
except KeyboardInterrupt:
    stop_event.set() 
    print("Stopping energy emulator..."); # Signal the thread to stop

generator_thread.join(); consumer_thread.join() # Finish child threads - producer + consumer
print("Process concluded.")