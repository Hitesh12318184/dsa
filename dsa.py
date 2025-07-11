import pandas as pd # type: ignore #ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.cluster import DBSCAN, KMeans #type: ignore
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #type: ignore
from sklearn.model_selection import train_test_split, cross_val_score #type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder #type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #type: ignore

# Geospatial Libraries
import folium #ignore #type: ignore
from folium.plugins import HeatMap, MarkerCluster #ignore #type: ignore
from geopy.distance import geodesic #ignore#type: ignore
from scipy.spatial.distance import cdist #ignore#type: ignore
from sklearn.neighbors import NearestNeighbors #ignore#type: ignore

# Time Series Libraries
from statsmodels.tsa.seasonal import seasonal_decompose #ignore#type: ignore
from statsmodels.tsa.arima.model import ARIMA #ignore#type: ignore
from statsmodels.tsa.stattools import adfuller #ignore#type: ignore

# Data Structures
from collections import defaultdict, Counter
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# ==================== DATA STRUCTURES ====================

@dataclass
class CrimeRecord:
    """Data structure for individual crime records"""
    id: str
    date: datetime
    latitude: float
    longitude: float
    crime_type: str
    description: str
    arrest: bool
    domestic: bool
    district: int
    ward: int

class CrimeHashTable:
    """Hash table for efficient crime data storage and retrieval"""
    def __init__(self, size=10000):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def insert(self, key, value):
        index = self._hash(key)
        self.table[index].append((key, value))
    
    def get(self, key):
        index = self._hash(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None
    
    def get_all_by_type(self, crime_type):
        """Get all crimes of a specific type"""
        crimes = []
        for bucket in self.table:
            for key, value in bucket:
                if hasattr(value, 'crime_type') and value.crime_type == crime_type:
                    crimes.append(value)
        return crimes

class SpatialIndex:
    """KD-Tree based spatial index for efficient location queries"""
    def __init__(self, points):
        self.points = np.array(points)
        self.tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        self.tree.fit(self.points)
    
    def find_nearest(self, point, k=1):
        distances, indices = self.tree.kneighbors([point], n_neighbors=k)
        return distances[0], indices[0]
    
    def range_query(self, point, radius):
        """Find all points within radius of given point"""
        indices = self.tree.radius_neighbors([point], radius=radius)[1][0]
        return self.points[indices]

class PriorityQueue:
    """Priority queue for crime severity ranking"""
    def __init__(self):
        self.heap = []
    
    def push(self, item, priority):
        heapq.heappush(self.heap, (-priority, item))
    
    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        return None
    
    def is_empty(self):
        return len(self.heap) == 0

# ==================== CRIME DATA GENERATOR ====================

class CrimeDataGenerator:
    """Generate synthetic crime data for testing"""
    
    def __init__(self, n_records=10000):
        self.n_records = n_records
        self.crime_types = ['THEFT', 'BURGLARY', 'ASSAULT', 'ROBBERY', 'VANDALISM', 
                           'DRUG_OFFENSE', 'FRAUD', 'VEHICLE_THEFT', 'HOMICIDE']
        self.crime_weights = {
            'THEFT': 0.25, 'BURGLARY': 0.15, 'ASSAULT': 0.15, 'ROBBERY': 0.10,
            'VANDALISM': 0.10, 'DRUG_OFFENSE': 0.08, 'FRAUD': 0.07,
            'VEHICLE_THEFT': 0.07, 'HOMICIDE': 0.03
        }
        
        # Chicago-like coordinates
        self.lat_range = (41.6, 42.0)
        self.lon_range = (-87.9, -87.5)
    
    def generate_synthetic_data(self):
        """Generate synthetic crime dataset"""
        np.random.seed(42)
        
        data = []
        for i in range(self.n_records):
            # Generate temporal patterns (more crimes at night and weekends)
            base_date = datetime(2020, 1, 1)
            days_offset = np.random.randint(0, 1460)  # 4 years
            hour = np.random.choice(24, p=self._get_hourly_distribution())
            
            crime_date = base_date + timedelta(days=days_offset, hours=hour)
            
            # Generate spatial clusters (hotspots)
            cluster_center = np.random.choice(5)
            cluster_centers = [
                (41.85, -87.65), (41.75, -87.75), (41.90, -87.60),
                (41.80, -87.70), (41.88, -87.63)
            ]
            
            center_lat, center_lon = cluster_centers[cluster_center]
            latitude = np.random.normal(center_lat, 0.02)
            longitude = np.random.normal(center_lon, 0.02)
            
            # Ensure coordinates are within bounds
            latitude = np.clip(latitude, *self.lat_range)
            longitude = np.clip(longitude, *self.lon_range)
            
            # Generate crime type based on weights
            crime_type = np.random.choice(list(self.crime_weights.keys()), 
                                        p=list(self.crime_weights.values()))
            
            # Generate other attributes
            arrest = np.random.choice([True, False], p=[0.2, 0.8])
            domestic = np.random.choice([True, False], p=[0.1, 0.9])
            district = np.random.randint(1, 26)
            ward = np.random.randint(1, 51)
            
            record = CrimeRecord(
                id=f"CR{i:06d}",
                date=crime_date,
                latitude=latitude,
                longitude=longitude,
                crime_type=crime_type,
                description=f"{crime_type} incident",
                arrest=arrest,
                domestic=domestic,
                district=district,
                ward=ward
            )
            data.append(record)
        
        return data
    
    def _get_hourly_distribution(self):
        """Generate realistic hourly crime distribution"""
        # More crimes during evening and night hours
        base_prob = np.ones(24) * 0.03
        base_prob[18:24] = 0.06  # Evening
        base_prob[0:6] = 0.04    # Late night/early morning
        base_prob[12:18] = 0.045 # Afternoon
        return base_prob / base_prob.sum()

# ==================== CRIME PATTERN DETECTOR ====================

class CrimePatternDetector:
    """Main class for crime pattern detection and analysis"""
    
    def __init__(self):
        self.crime_data = None
        self.crime_hash_table = CrimeHashTable()
        self.spatial_index = None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
    def load_data(self, crime_records):
        """Load crime data into the system"""
        self.crime_data = crime_records
        
        # Populate hash table
        for crime in crime_records:
            self.crime_hash_table.insert(crime.id, crime)
        
        # Create spatial index
        coordinates = [(crime.latitude, crime.longitude) for crime in crime_records]
        self.spatial_index = SpatialIndex(coordinates)
        
        print(f"Loaded {len(crime_records)} crime records")
    
    def preprocess_data(self):
        """Convert crime records to DataFrame and preprocess"""
        data = []
        for crime in self.crime_data:
            data.append({
                'id': crime.id,
                'date': crime.date,
                'latitude': crime.latitude,
                'longitude': crime.longitude,
                'crime_type': crime.crime_type,
                'arrest': crime.arrest,
                'domestic': crime.domestic,
                'district': crime.district,
                'ward': crime.ward,
                'hour': crime.date.hour,
                'day_of_week': crime.date.weekday(),
                'month': crime.date.month,
                'year': crime.date.year
            })
        
        df = pd.DataFrame(data)
        
        # Feature engineering
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['season'] = df['month'].apply(self._get_season)
        
        return df
    
    def _get_season(self, month):
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def detect_hotspots(self, eps=0.01, min_samples=5):
        """Detect crime hotspots using DBSCAN clustering"""
        coordinates = np.array([(crime.latitude, crime.longitude) 
                               for crime in self.crime_data])
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(coordinates)
        
        # Analyze clusters
        hotspots = []
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Noise points
                continue
                
            cluster_points = coordinates[clusters == cluster_id]
            center = np.mean(cluster_points, axis=0)
            size = len(cluster_points)
            
            hotspots.append({
                'cluster_id': cluster_id,
                'center_lat': center[0],
                'center_lon': center[1],
                'crime_count': size,
                'density': size / (np.pi * (eps ** 2))
            })
        
        return sorted(hotspots, key=lambda x: x['crime_count'], reverse=True)
    
    def analyze_temporal_patterns(self):
        """Analyze temporal crime patterns"""
        df = self.preprocess_data()
        
        # Hourly patterns
        hourly_counts = df.groupby('hour').size()
        
        # Daily patterns
        daily_counts = df.groupby('day_of_week').size()
        
        # Monthly patterns
        monthly_counts = df.groupby('month').size()
        
        # Seasonal patterns
        seasonal_counts = df.groupby('season').size()
        
        return {
            'hourly': hourly_counts,
            'daily': daily_counts,
            'monthly': monthly_counts,
            'seasonal': seasonal_counts
        }
    
    def build_crime_prediction_model(self):
        """Build machine learning model for crime prediction"""
        df = self.preprocess_data()
        
        # Prepare features
        feature_columns = ['latitude', 'longitude', 'hour', 'day_of_week', 
                          'month', 'district', 'ward', 'is_weekend', 'is_night']
        
        X = df[feature_columns]
        y = df['crime_type']
        
        # Encode categorical variables
        self.label_encoders['crime_type'] = LabelEncoder()
        y_encoded = self.label_encoders['crime_type'].fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        # Train Random Forest model
        self.models['crime_classifier'] = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.models['crime_classifier'].fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.models['crime_classifier'].predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Crime Type Prediction Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoders['crime_type'].classes_))
        
        return {
            'accuracy': accuracy,
            'model': self.models['crime_classifier'],
            'scaler': self.scalers['features']
        }
    
    def predict_crime_risk(self, latitude, longitude, hour, day_of_week, 
                          month, district, ward):
        """Predict crime risk for given location and time"""
        if 'crime_classifier' not in self.models:
            raise ValueError("Model not trained. Call build_crime_prediction_model() first.")
        
        # Prepare input
        is_weekend = 1 if day_of_week in [5, 6] else 0
        is_night = 1 if hour >= 22 or hour <= 6 else 0
        
        features = np.array([[latitude, longitude, hour, day_of_week, 
                             month, district, ward, is_weekend, is_night]])
        
        # Scale features
        features_scaled = self.scalers['features'].transform(features)
        
        # Predict
        probabilities = self.models['crime_classifier'].predict_proba(features_scaled)[0]
        predicted_class = self.models['crime_classifier'].predict(features_scaled)[0]
        
        # Decode prediction
        crime_types = self.label_encoders['crime_type'].classes_
        predicted_crime_type = crime_types[predicted_class]
        
        # Create risk assessment
        risk_scores = dict(zip(crime_types, probabilities))
        
        return {
            'predicted_crime_type': predicted_crime_type,
            'confidence': max(probabilities),
            'risk_scores': risk_scores,
            'overall_risk': 'High' if max(probabilities) > 0.3 else 'Medium' if max(probabilities) > 0.15 else 'Low'
        }
    
    def optimize_patrol_routes(self, hotspots, n_patrols=5):
        """Optimize patrol routes using hotspot data"""
        # Use K-means to assign hotspots to patrol units
        if len(hotspots) < n_patrols:
            n_patrols = len(hotspots)
        
        coordinates = np.array([(h['center_lat'], h['center_lon']) for h in hotspots])
        weights = np.array([h['crime_count'] for h in hotspots])
        
        # Weighted K-means clustering
        kmeans = KMeans(n_clusters=n_patrols, random_state=42)
        patrol_assignments = kmeans.fit_predict(coordinates, sample_weight=weights)
        
        # Create patrol routes
        patrol_routes = []
        for i in range(n_patrols):
            assigned_hotspots = [hotspots[j] for j in range(len(hotspots)) 
                               if patrol_assignments[j] == i]
            
            if assigned_hotspots:
                # Calculate route efficiency
                total_crimes = sum(h['crime_count'] for h in assigned_hotspots)
                avg_lat = np.mean([h['center_lat'] for h in assigned_hotspots])
                avg_lon = np.mean([h['center_lon'] for h in assigned_hotspots])
                
                patrol_routes.append({
                    'patrol_id': i + 1,
                    'hotspots': assigned_hotspots,
                    'total_crimes': total_crimes,
                    'center_lat': avg_lat,
                    'center_lon': avg_lon,
                    'priority': 'High' if total_crimes > 100 else 'Medium' if total_crimes > 50 else 'Low'
                })
        
        return sorted(patrol_routes, key=lambda x: x['total_crimes'], reverse=True)
    
    def generate_crime_forecast(self, days_ahead=30):
        """Generate crime forecast using time series analysis"""
        df = self.preprocess_data()
        
        # Aggregate daily crime counts
        daily_counts = df.groupby(df['date'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'crime_count']
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        daily_counts = daily_counts.set_index('date').asfreq('D', fill_value=0)
        
        # Simple moving average forecast
        window = 7
        forecast = []
        last_values = daily_counts['crime_count'].tail(window).values
        
        for i in range(days_ahead):
            next_value = np.mean(last_values)
            forecast.append(next_value)
            last_values = np.append(last_values[1:], next_value)
        
        # Create forecast dates
        last_date = daily_counts.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=days_ahead, freq='D')
        
        return pd.DataFrame({
            'date': forecast_dates,
            'predicted_crimes': forecast
        })
    
    def visualize_crime_patterns(self, save_plots=True):
        """Create visualizations of crime patterns"""
        df = self.preprocess_data()
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Hourly crime distribution
        hourly_counts = df.groupby('hour').size()
        axes[0, 0].bar(hourly_counts.index, hourly_counts.values, color='skyblue')
        axes[0, 0].set_title('Crime Distribution by Hour of Day')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Number of Crimes')
        
        # 2. Daily crime distribution
        daily_counts = df.groupby('day_of_week').size()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(range(7), daily_counts.values, color='lightcoral')
        axes[0, 1].set_title('Crime Distribution by Day of Week')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Number of Crimes')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(days)
        
        # 3. Crime type distribution
        crime_type_counts = df['crime_type'].value_counts().head(8)
        axes[1, 0].barh(range(len(crime_type_counts)), crime_type_counts.values, 
                       color='lightgreen')
        axes[1, 0].set_title('Top Crime Types')
        axes[1, 0].set_xlabel('Number of Crimes')
        axes[1, 0].set_yticks(range(len(crime_type_counts)))
        axes[1, 0].set_yticklabels(crime_type_counts.index)
        
        # 4. Monthly crime trends
        monthly_counts = df.groupby('month').size()
        axes[1, 1].plot(monthly_counts.index, monthly_counts.values, 
                       marker='o', linewidth=2, color='purple')
        axes[1, 1].set_title('Crime Distribution by Month')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Number of Crimes')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('crime_patterns_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def create_crime_heatmap(self, save_map=True):
        """Create interactive heatmap of crime locations"""
        df = self.preprocess_data()
        
        # Create base map centered on data
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Add heatmap layer
        heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
        HeatMap(heat_data, radius=15, blur=15, max_zoom=1).add_to(m)
        
        # Add hotspot markers
        hotspots = self.detect_hotspots()
        for hotspot in hotspots[:10]:  # Top 10 hotspots
            folium.CircleMarker(
                location=[hotspot['center_lat'], hotspot['center_lon']],
                radius=hotspot['crime_count'] / 10,
                popup=f"Hotspot {hotspot['cluster_id']}<br>Crimes: {hotspot['crime_count']}",
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.6
            ).add_to(m)
        
        if save_map:
            m.save('crime_heatmap.html')
        
        return m
    
    def generate_report(self):
        """Generate comprehensive crime analysis report"""
        df = self.preprocess_data()
        hotspots = self.detect_hotspots()
        temporal_patterns = self.analyze_temporal_patterns()
        
        report = {
            'summary': {
                'total_crimes': len(df),
                'date_range': f"{df['date'].min()} to {df['date'].max()}",
                'most_common_crime': df['crime_type'].mode()[0],
                'hotspots_detected': len(hotspots),
                'peak_hour': temporal_patterns['hourly'].idxmax(),
                'peak_day': temporal_patterns['daily'].idxmax()
            },
            'hotspots': hotspots[:5],  # Top 5 hotspots
            'temporal_patterns': temporal_patterns,
            'recommendations': self._generate_recommendations(df, hotspots)
        }
        
        return report
    
    def _generate_recommendations(self, df, hotspots):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Hotspot recommendations
        if hotspots:
            top_hotspot = hotspots[0]
            recommendations.append(
                f"Deploy additional patrols to hotspot at ({top_hotspot['center_lat']:.3f}, "
                f"{top_hotspot['center_lon']:.3f}) with {top_hotspot['crime_count']} crimes"
            )
        
        # Temporal recommendations
        peak_hour = df.groupby('hour').size().idxmax()
        if peak_hour >= 18:
            recommendations.append(f"Increase evening patrols around {peak_hour}:00")
        
        # Crime type recommendations
        most_common_crime = df['crime_type'].mode()[0]
        recommendations.append(f"Focus prevention efforts on {most_common_crime}")
        
        return recommendations

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    print("Crime Pattern Detection System")
    print("=" * 50)
    
    # Initialize system
    detector = CrimePatternDetector()
    
    # Generate synthetic data
    print("1. Generating synthetic crime data...")
    generator = CrimeDataGenerator(n_records=5000)
    crime_records = generator.generate_synthetic_data()
    
    # Load data
    print("2. Loading data into system...")
    detector.load_data(crime_records)
    
    # Detect hotspots
    print("3. Detecting crime hotspots...")
    hotspots = detector.detect_hotspots()
    print(f"   Found {len(hotspots)} hotspots")
    
    # Analyze temporal patterns
    print("4. Analyzing temporal patterns...")
    temporal_patterns = detector.analyze_temporal_patterns()
    
    # Build prediction model
    print("5. Building crime prediction model...")
    model_results = detector.build_crime_prediction_model()
    
    # Generate patrol routes
    print("6. Optimizing patrol routes...")
    patrol_routes = detector.optimize_patrol_routes(hotspots)
    print(f"   Generated {len(patrol_routes)} patrol routes")
    
    # Create forecast
    print("7. Generating crime forecast...")
    forecast = detector.generate_crime_forecast(days_ahead=30)
    
    # Make sample prediction
    print("8. Making sample risk prediction...")
    risk_prediction = detector.predict_crime_risk(
        latitude=41.85, longitude=-87.65, hour=22, day_of_week=5,
        month=8, district=12, ward=25
    )
    
    # Generate visualizations
    print("9. Creating visualizations...")
    detector.visualize_crime_patterns()
    
    # Create heatmap
    print("10. Creating crime heatmap...")
    crime_map = detector.create_crime_heatmap()
    
    # Generate report
    print("11. Generating comprehensive report...")
    report = detector.generate_report()
    
    # Print results
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    print(f"\nTotal Crimes Analyzed: {report['summary']['total_crimes']}")
    print(f"Date Range: {report['summary']['date_range']}")
    print(f"Most Common Crime Type: {report['summary']['most_common_crime']}")
    print(f"Peak Hour: {report['summary']['peak_hour']}:00")
    print(f"Hotspots Detected: {report['summary']['hotspots_detected']}")
    
    print(f"\nTop 3 Hotspots:")
    for i, hotspot in enumerate(report['hotspots'][:3]):
        print(f"  {i+1}. Location: ({hotspot['center_lat']:.3f}, {hotspot['center_lon']:.3f})")
        print(f"     Crime Count: {hotspot['crime_count']}")
    
    print(f"\nSample Risk Prediction:")
    print(f"  Location: (41.85, -87.65) at 22:00 on Friday")
    print(f"  Predicted Crime Type: {risk_prediction['predicted_crime_type']}")
    print(f"  Confidence: {risk_prediction['confidence']:.3f}")
    print(f"  Overall Risk: {risk_prediction['overall_risk']}")
    
    print(f"\nTop 3 Patrol Routes:")
    for i, route in enumerate(patrol_routes[:3]):
        print(f"  Route {i+1}: {route['total_crimes']} crimes, Priority: {route['priority']}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report['recommendations']):
        print(f"  {i+1}. {rec}")
    
    print(f"\nForecast (Next 7 days average): {forecast['predicted_crimes'][:7].mean():.1f} crimes/day")
    
    print("\n" + "=" * 50)
    print("Analysis complete! Check generated files:")
    print("- crime_patterns_analysis.png (visualization)")
    print("- crime_heatmap.html (interactive map)")
    print("=" * 50)

if __name__ == "__main__":
    main()