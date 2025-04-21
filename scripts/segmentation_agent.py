import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from datetime import datetime
from openai import OpenAI
from agent import Agent
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class SegmentationAgent(Agent):
    """
    Agent that uses K-means clustering and DeepSeek LLM to segment customers based on RFM data
    """
    name = "Segmentation Agent"
    color = Agent.CYAN
    MODEL = "deepseek-r1:1.5b"
    
    def __init__(self, output_dir="output"):
        """
        Set up this instance by connecting to DeepSeek and initializing segmentation state
        """
        self.log("Initializing Segmentation Agent")
        
        # Store output directory
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up the DeepSeek client via Ollama
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        # Segmentation state
        self.rfm_data = None
        self.segments = None
        self.segment_descriptions = None
        self.segment_strategies = None
        self.kmeans_model = None
        self.scaler = None
        
        self.log(f"Segmentation Agent is ready with {self.MODEL}")
    
    def load_data(self, data_path: str) -> bool:
        """
        Load the RFM data for segmentation
        """
        self.log(f"Loading data from {data_path}")
        
        try:
            # Determine file type and load accordingly
            if data_path.endswith('.parquet'):
                self.rfm_data = pd.read_parquet(data_path)
            elif data_path.endswith('.csv'):
                self.rfm_data = pd.read_csv(data_path)
            else:
                self.log(f"Unsupported file format: {data_path}")
                return False
            
            # Ensure required columns exist
            required_columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
            for col in required_columns:
                if col not in self.rfm_data.columns:
                    self.log(f"Missing required column: {col}")
                    return False
            
            self.log(f"Data loaded successfully with {len(self.rfm_data)} customers")
            return True
        
        except Exception as e:
            self.log(f"Error loading data: {str(e)}")
            return False
    
    def _get_rfm_summary(self) -> Dict[str, Any]:
        """Analyze RFM data and create a summary for the LLM"""
        summary = {}
        
        # Calculate basic statistics
        for metric in ['Recency', 'Frequency', 'Monetary']:
            summary[metric] = {
                "min": float(self.rfm_data[metric].min()),
                "max": float(self.rfm_data[metric].max()),
                "mean": float(self.rfm_data[metric].mean()),
                "median": float(self.rfm_data[metric].median()),
                "p25": float(self.rfm_data[metric].quantile(0.25)),
                "p75": float(self.rfm_data[metric].quantile(0.75))
            }
        
        # Calculate correlations
        corr_matrix = self.rfm_data[['Recency', 'Frequency', 'Monetary']].corr()
        summary["correlations"] = {
            "Recency_Frequency": float(corr_matrix.loc['Recency', 'Frequency']),
            "Recency_Monetary": float(corr_matrix.loc['Recency', 'Monetary']),
            "Frequency_Monetary": float(corr_matrix.loc['Frequency', 'Monetary'])
        }
        
        return summary
    
    def messages_for_cluster_interpretation(self, clusters: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create the message list for generating cluster interpretations
        """
        system_message = """You are an expert in customer segmentation and marketing analytics.
        Your task is to interpret K-means clusters from RFM (Recency, Frequency, Monetary) data 
        and provide meaningful segment names and descriptions."""
        
        user_message = f"""
        I have performed K-means clustering on customer RFM data and need your help interpreting the clusters.
        
        Here are the clusters with their average RFM values and other statistics:
        {json.dumps(clusters, indent=2)}
        
        For each cluster, please:
        1. Provide a descriptive name that reflects the customer behavior (like "Champions", "At Risk", etc.)
        2. Write a detailed description of the customer behavior and characteristics
        3. Suggest specific marketing strategies tailored to this customer segment
        
        Format your response as a JSON object with the following structure:
        {{
            "renamed_clusters": {{
                "0": "Descriptive name for cluster 0",
                "1": "Descriptive name for cluster 1",
                ...
            }},
            "descriptions": {{
                "0": "Detailed description for cluster 0...",
                "1": "Detailed description for cluster 1...",
                ...
            }},
            "strategies": {{
                "0": "Marketing strategies for cluster 0...",
                "1": "Marketing strategies for cluster 1...",
                ...
            }}
        }}
        
        Be specific and insightful in your analysis. Use the RFM values to inform your interpretation.
        Return only the JSON object.
        """
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "{"}
        ]
    
    def messages_for_report(self, segments: Dict[str, Any], descriptions: Dict[str, str], strategies: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Create the message list for generating a report
        """
        system_message = """You are a marketing analytics expert who creates comprehensive business reports.
        Create a detailed customer segmentation report in Markdown format."""
        
        user_message = f"""
        Create a comprehensive customer segmentation report based on the provided segmentation data. The report should include:

        1. Executive Summary
        2. Introduction and Methodology (mention that K-means clustering was used on RFM data)
        3. Overview of Customer Segments
        4. Detailed Analysis of Each Segment
        5. Marketing Recommendations
        6. Conclusion

        Here's the segmentation data:
        {json.dumps(segments)}

        Additional insights about each segment:
        Descriptions:
        {json.dumps(descriptions)}

        Marketing Strategies:
        {json.dumps(strategies)}

        Format the report in Markdown with proper headings, bullet points, and any tables/charts you think would be useful (described in Markdown).
        Make the report professional, insightful, and actionable for business stakeholders.
        """
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def extract_json(self, content: str) -> Any:
        """Extract JSON from LLM response"""
        try:
            # Try to parse as-is first
            return json.loads(content)
        except:
            # Try to extract JSON if embedded in other text
            try:
                # For objects
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                
                # For arrays
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                
                return None
            except:
                return None
    
    def _find_optimal_k(self, X_scaled, max_k=10) -> int:
        """Find optimal number of clusters using silhouette score"""
        self.log("Finding optimal number of clusters")
        
        if len(X_scaled) < 3:
            return 3  # Minimum clusters for meaningful segmentation
        
        max_k = min(max_k, len(X_scaled) - 1)
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            self.log(f"K={k}, Silhouette Score={silhouette_avg:.4f}")
        
        # Find the best k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        self.log(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def perform_segmentation(self, num_segments: int = 0) -> Dict[str, Any]:
        """
        Perform customer segmentation using K-means clustering
        """
        self.log(f"Starting K-means segmentation")
        
        if self.rfm_data is None:
            self.log("No data loaded. Please load data first.")
            return {}
        
        # Get RFM features for clustering
        X = self.rfm_data[['Recency', 'Frequency', 'Monetary']].copy()
        
        # Scale features - switch StandardScaler to RobustScaler for better handling of outliers
        self.scaler = RobustScaler()  # Change to RobustScaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Force more clusters if optimal number is too small
        if num_segments <= 0:
            num_segments = self._find_optimal_k(X_scaled)
            num_segments = max(num_segments, 4)  # Force at least 4 clusters
        
        # Perform K-means clustering with better initialization
        self.log(f"Performing K-means clustering with {num_segments} clusters")
        self.kmeans_model = KMeans(
            n_clusters=num_segments,
            random_state=42,
            n_init=15,  # Increase number of initializations
            init='k-means++',
            max_iter=500  # Allow more iterations for convergence
        )
        self.rfm_data['segment_id'] = self.kmeans_model.fit_predict(X_scaled)
        
        # Debug clustering results
        unique, counts = np.unique(self.rfm_data['segment_id'], return_counts=True)
        cluster_distribution = dict(zip(unique, counts))
        self.log(f"Cluster distribution: {cluster_distribution}")
        
        # Create segment information
        segments = {}
        for i in range(num_segments):
            segment_mask = self.rfm_data['segment_id'] == i
            segment_data = self.rfm_data[segment_mask]
            
            # Add temporary generic names
            self.rfm_data.loc[segment_mask, 'segment_name'] = f"Cluster {i}"
            
            segments[str(i)] = {
                "segment_id": i,
                "segment_name": f"Cluster {i}",
                "count": int(segment_mask.sum()),
                "percentage": float(segment_mask.sum() / len(self.rfm_data) * 100),
                "center_recency": float(self.kmeans_model.cluster_centers_[i, 0]),
                "center_frequency": float(self.kmeans_model.cluster_centers_[i, 1]),
                "center_monetary": float(self.kmeans_model.cluster_centers_[i, 2]),
                "avg_recency": float(segment_data['Recency'].mean()),
                "avg_frequency": float(segment_data['Frequency'].mean()),
                "avg_monetary": float(segment_data['Monetary'].mean()),
                "total_monetary": float(segment_data['Monetary'].sum()),
                "std_recency": float(segment_data['Recency'].std()),
                "std_frequency": float(segment_data['Frequency'].std()),
                "std_monetary": float(segment_data['Monetary'].std())
            }
        
        self.segments = segments
        
        # Use LLM to interpret the clusters
        try:
            self.log(f"Calling {self.MODEL} to interpret clusters")
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages_for_cluster_interpretation(self.segments),
                max_tokens=2048
            )
            
            content = response.choices[0].message.content
            cluster_insights = self.extract_json(content)
            
            if (not cluster_insights or 
                    "renamed_clusters" not in cluster_insights or
                    "descriptions" not in cluster_insights or
                    "strategies" not in cluster_insights):
                self.log("Failed to parse cluster insights from LLM response")
                insights = self._get_fallback_insights(self.segments)
                self.segment_descriptions = insights["descriptions"]
                self.segment_strategies = insights["strategies"]
                
                # Use fallback names
                for segment_id, segment in self.segments.items():
                    segment["segment_name"] = f"Segment {segment_id}"
            else:
                # Update segment names
                for segment_id, new_name in cluster_insights["renamed_clusters"].items():
                    if segment_id in self.segments:
                        self.segments[segment_id]["segment_name"] = new_name
                        # Update in the dataframe too
                        self.rfm_data.loc[self.rfm_data['segment_id'] == int(segment_id), 'segment_name'] = new_name
                
                self.segment_descriptions = cluster_insights["descriptions"]
                self.segment_strategies = cluster_insights["strategies"]
        except Exception as e:
            self.log(f"Error using LLM for cluster interpretation: {str(e)}")
            insights = self._get_fallback_insights(self.segments)
            self.segment_descriptions = insights["descriptions"]
            self.segment_strategies = insights["strategies"]
        
        # Compile results
        results = {
            "num_customers": len(self.rfm_data),
            "num_segments": len(self.segments),
            "segments": self.segments,
            "segment_descriptions": self.segment_descriptions,
            "segment_strategies": self.segment_strategies,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save results
        results_file = os.path.join(self.output_dir, "kmeans_segmentation_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        self.log(f"Segmentation complete - created {len(self.segments)} segments with K-means")
        return results
    
    def _get_fallback_insights(self, segments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback insights if LLM fails"""
        descriptions = {}
        strategies = {}
        
        for segment_id, segment in segments.items():
            # Create basic descriptions
            descriptions[segment_id] = (
                f"Cluster {segment_id}: This segment contains {segment['count']} customers ({segment['percentage']:.1f}% of total). "
                f"On average, they made their last purchase {segment['avg_recency']:.1f} days ago, "
                f"have made {segment['avg_frequency']:.1f} purchases, and "
                f"spent ${segment['avg_monetary']:.2f} per customer."
            )
            
            # Create basic strategies based on RFM values
            recency = segment['avg_recency']
            frequency = segment['avg_frequency']
            monetary = segment['avg_monetary']
            
            if recency < segments["0"]["avg_recency"] and frequency > segments["0"]["avg_frequency"]:
                strategies[segment_id] = (
                    "These are likely your best customers. Reward them with loyalty programs, exclusive offers, "
                    "and early access to new products. Focus on building relationships and turning them into brand advocates."
                )
            elif recency > segments["0"]["avg_recency"] and frequency > segments["0"]["avg_frequency"]:
                strategies[segment_id] = (
                    "These customers used to shop frequently but haven't returned recently. Implement a win-back campaign "
                    "with personalized offers based on their past purchases."
                )
            elif recency < segments["0"]["avg_recency"] and frequency < segments["0"]["avg_frequency"]:
                strategies[segment_id] = (
                    "These are likely new customers. Focus on onboarding and education about your products and services. "
                    "Cross-sell related products to increase their engagement."
                )
            else:
                strategies[segment_id] = (
                    "Develop personalized communication and offers based on their purchasing patterns. "
                    "Analyze their behavior to find opportunities to increase engagement and spending."
                )
        
        return {
            "descriptions": descriptions,
            "strategies": strategies
        }
    
    def generate_report(self, output_file: str = "customer_segmentation_report.md") -> str:
        """
        Generate a comprehensive customer segmentation report
        """
        self.log(f"Generating customer segmentation report")
        
        if self.segments is None:
            self.log("No segmentation results. Please perform segmentation first.")
            return ""
        
        # Call DeepSeek to generate the report
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages_for_report(
                    self.segments, 
                    self.segment_descriptions,
                    self.segment_strategies
                ),
                max_tokens=4096
            )
            
            report_content = response.choices[0].message.content.strip()
        except Exception as e:
            self.log(f"Error generating report: {str(e)}")
            report_content = self._generate_fallback_report()
        
        # Save the report
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, "w") as f:
            f.write(report_content)
        
        self.log(f"Report saved to {output_path}")
        return output_path
    
    def _generate_fallback_report(self) -> str:
        """Generate a simple fallback report if LLM fails"""
        report = []
        report.append("# Customer Segmentation Report (K-means Clustering)\n")
        report.append("## Executive Summary\n")
        report.append("This report presents a customer segmentation analysis based on K-means clustering of RFM (Recency, Frequency, Monetary) data.\n")
        
        report.append("## Overview of Customer Segments\n")
        
        segment_table = "| Segment ID | Segment Name | Customer Count | Percentage | Avg. Recency | Avg. Frequency | Avg. Monetary |\n"
        segment_table += "|------------|--------------|---------------|------------|-------------|---------------|---------------|\n"
        
        for segment_id, segment in self.segments.items():
            segment_table += f"| {segment['segment_id']} | {segment['segment_name']} | {segment['count']} | {segment['percentage']:.1f}% | {segment['avg_recency']:.1f} | {segment['avg_frequency']:.1f} | ${segment['avg_monetary']:.2f} |\n"
        
        report.append(segment_table)
        
        report.append("\n## Detailed Analysis of Each Segment\n")
        
        for segment_id, segment in self.segments.items():
            report.append(f"### {segment['segment_name']}\n")
            
            if segment_id in self.segment_descriptions:
                report.append(f"{self.segment_descriptions[segment_id]}\n")
            else:
                report.append(f"- Customer count: {segment['count']} ({segment['percentage']:.1f}%)\n")
                report.append(f"- Average recency: {segment['avg_recency']:.1f} days\n")
                report.append(f"- Average frequency: {segment['avg_frequency']:.1f} purchases\n")
                report.append(f"- Average monetary value: ${segment['avg_monetary']:.2f}\n")
            
            report.append("\n**Marketing Strategies:**\n")
            
            if segment_id in self.segment_strategies:
                report.append(f"{self.segment_strategies[segment_id]}\n")
            
            report.append("\n")
        
        report.append("## Conclusion\n")
        report.append("This segmentation using K-means clustering has identified distinct customer groups that can be targeted with specific marketing strategies.\n")
        
        return "\n".join(report)
    
    def plot_segments(self, output_file: str = "segment_visualization.png") -> str:
        """
        Generate a visualization of the customer segments
        """
        self.log("Generating segment visualization")
        
        if self.rfm_data is None or 'segment_name' not in self.rfm_data.columns:
            self.log("No segmentation results. Please perform segmentation first.")
            return ""
        
        # Create a figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Pie chart of segment sizes
        segment_counts = self.rfm_data['segment_name'].value_counts()
        axes[0, 0].pie(
            segment_counts, 
            labels=segment_counts.index, 
            autopct='%1.1f%%', 
            startangle=90,
            colors=sns.color_palette('tab10', len(segment_counts))
        )
        axes[0, 0].set_title('Customer Segment Distribution')
        
        # 2. Scatter plot of Recency vs Frequency
        sns.scatterplot(
            data=self.rfm_data,
            x='Recency',
            y='Frequency',
            hue='segment_name',
            palette='tab10',
            alpha=0.7,
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Recency vs Frequency by Segment')
        
        # 3. Scatter plot of Frequency vs Monetary
        sns.scatterplot(
            data=self.rfm_data,
            x='Frequency',
            y='Monetary',
            hue='segment_name',
            palette='tab10',
            alpha=0.7,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Frequency vs Monetary by Segment')
        
        # 4. PCA visualization if more than 2 segments
        if len(self.segments) > 2:
            # Apply PCA for visualization
            X = self.rfm_data[['Recency', 'Frequency', 'Monetary']]
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.scaler.transform(X))
            
            # Create a dataframe with PCA results
            pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
            pca_df['segment_name'] = self.rfm_data['segment_name'].values
            
            # Plot PCA results
            sns.scatterplot(
                data=pca_df,
                x='PC1',
                y='PC2',
                hue='segment_name',
                palette='tab10',
                alpha=0.7,
                ax=axes[1, 1]
            )
            var_explained = pca.explained_variance_ratio_ * 100
            axes[1, 1].set_title(f'PCA Visualization (Explained Variance: {var_explained[0]:.1f}%, {var_explained[1]:.1f}%)')
        else:
            # Bar chart of average metrics by segment
            segment_avgs = self.rfm_data.groupby('segment_name')[['Recency', 'Frequency', 'Monetary']].mean()
            
            # Normalize the metrics for better visualization
            segment_avgs_norm = segment_avgs.copy()
            for col in segment_avgs_norm.columns:
                if col == 'Recency':
                    # For Recency, lower is better, so invert the normalization
                    segment_avgs_norm[col] = 1 - (segment_avgs_norm[col] / segment_avgs_norm[col].max())
                else:
                    segment_avgs_norm[col] = segment_avgs_norm[col] / segment_avgs_norm[col].max()
            
            segment_avgs_norm.plot(
                kind='bar',
                ax=axes[1, 1],
                color=['#ff9999', '#66b3ff', '#99ff99'],
                rot=45
            )
            axes[1, 1].set_title('Normalized RFM Metrics by Segment')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].legend(title='Metric')
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path)
        plt.close()
        
        self.log(f"Visualization saved to {output_path}")
        return output_path
    
    def segment_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Segment a single customer based on their RFM data
        """
        self.log(f"Segmenting individual customer")
        
        if self.kmeans_model is None or self.scaler is None:
            self.log("No segmentation model. Please perform segmentation first.")
            return {"error": "No segmentation model available"}
        
        # Validate input
        required_fields = ['Recency', 'Frequency', 'Monetary']
        for field in required_fields:
            if field not in customer_data:
                self.log(f"Missing required field: {field}")
                return {"error": f"Missing required field: {field}"}
        
        # Prepare the customer data for prediction
        customer_array = np.array([[
            customer_data['Recency'],
            customer_data['Frequency'],
            customer_data['Monetary']
        ]])
        
        # Scale the data
        customer_scaled = self.scaler.transform(customer_array)
        
        # Predict the segment
        segment_id = int(self.kmeans_model.predict(customer_scaled)[0])
        segment_id_str = str(segment_id)
        
        # Get segment details
        if segment_id_str in self.segments:
            segment = self.segments[segment_id_str]
            
            result = {
                "segment_id": segment_id,
                "segment_name": segment["segment_name"],
                "description": self.segment_descriptions.get(segment_id_str, ""),
                "recommended_strategy": self.segment_strategies.get(segment_id_str, "")
            }
            return result
        else:
            self.log(f"Unexpected segment ID: {segment_id}")
            return {
                "segment_id": segment_id,
                "segment_name": f"Cluster {segment_id}",
                "description": "Unknown segment",
                "recommended_strategy": "Further analysis needed"
            }