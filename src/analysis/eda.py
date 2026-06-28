"""
eda.py
──────
Exploratory Data Analysis for the IBM Telco Churn Dataset.
Generates publication-quality visualizations saved to dashboard_images/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import config

# ── Global Style ──────────────────────────────────────────────────────────────
plt.rcParams['figure.dpi']       = 150
plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['axes.spines.top']  = False
plt.rcParams['axes.spines.right']= False

COLORS  = ['#2ecc71', '#e74c3c']   # Green = No Churn, Red = Churn
IMG_DIR = config.ROOT_DIR / 'dashboard_images'
IMG_DIR.mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv(config.CLEAN_DATA_FILE)
    print(f"Loaded clean data: {df.shape}")
    return df

# ── Plot 1: Churn Distribution ────────────────────────────────────────────────
def plot_churn_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Customer Churn Distribution', fontsize=16, fontweight='bold')

    counts = df['Churn'].value_counts()
    labels = ['No Churn', 'Churned']

    # Bar chart
    axes[0].bar(labels, counts.values, color=COLORS, edgecolor='white', width=0.5)
    axes[0].set_title('Customer Count')
    axes[0].set_ylabel('Number of Customers')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=COLORS,
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Churn Rate')

    plt.tight_layout()
    plt.savefig(IMG_DIR / '01_churn_distribution.png', bbox_inches='tight')
    plt.close()
    print("Saved: 01_churn_distribution.png")

# ── Plot 2: Churn by Contract Type ────────────────────────────────────────────
def plot_churn_by_contract(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Churn Rate by Contract Type', fontsize=16, fontweight='bold')

    contract_churn = df.groupby('Contract')['Churn'].mean() * 100
    bars = ax.bar(contract_churn.index, contract_churn.values,
                  color=['#e74c3c', '#f39c12', '#2ecc71'],
                  edgecolor='white', width=0.5)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel('Churn Rate (%)')
    ax.set_xlabel('Contract Type')

    for bar, val in zip(bars, contract_churn.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(IMG_DIR / '02_churn_by_contract.png', bbox_inches='tight')
    plt.close()
    print("Saved: 02_churn_by_contract.png")

# ── Plot 3: Monthly Charges Distribution ──────────────────────────────────────
def plot_monthly_charges(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Monthly Charges Distribution by Churn Status',
                 fontsize=16, fontweight='bold')

    for churn_val, label, color in zip([0, 1], ['No Churn', 'Churned'], COLORS):
        subset = df[df['Churn'] == churn_val]['MonthlyCharges']
        ax.hist(subset, bins=40, alpha=0.6, color=color,
                label=f'{label} (mean: \${subset.mean():.0f})', edgecolor='white')

    ax.set_xlabel('Monthly Charges (\$)')
    ax.set_ylabel('Number of Customers')
    ax.legend()

    plt.tight_layout()
    plt.savefig(IMG_DIR / '03_monthly_charges.png', bbox_inches='tight')
    plt.close()
    print("Saved: 03_monthly_charges.png")

# ── Plot 4: Tenure Analysis ───────────────────────────────────────────────────
def plot_tenure_analysis(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Customer Tenure Distribution by Churn Status',
                 fontsize=16, fontweight='bold')

    for churn_val, label, color in zip([0, 1], ['No Churn', 'Churned'], COLORS):
        subset = df[df['Churn'] == churn_val]['tenure']
        ax.hist(subset, bins=36, alpha=0.6, color=color,
                label=f'{label} (mean: {subset.mean():.0f} months)',
                edgecolor='white')

    ax.set_xlabel('Tenure (Months)')
    ax.set_ylabel('Number of Customers')
    ax.legend()

    plt.tight_layout()
    plt.savefig(IMG_DIR / '04_tenure_analysis.png', bbox_inches='tight')
    plt.close()
    print("Saved: 04_tenure_analysis.png")

# ── Plot 5: Churn by Internet Service ─────────────────────────────────────────
def plot_churn_by_internet(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Churn Rate by Internet Service Type',
                 fontsize=16, fontweight='bold')

    internet_churn = df.groupby('InternetService')['Churn'].mean() * 100
    bars = ax.bar(internet_churn.index, internet_churn.values,
                  color=['#3498db', '#e74c3c', '#95a5a6'],
                  edgecolor='white', width=0.5)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel('Churn Rate (%)')
    ax.set_xlabel('Internet Service Type')

    for bar, val in zip(bars, internet_churn.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(IMG_DIR / '05_churn_by_internet.png', bbox_inches='tight')
    plt.close()
    print("Saved: 05_churn_by_internet.png")

# ── Plot 6: Correlation Heatmap ───────────────────────────────────────────────
def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Correlation Matrix — Numerical Features',
                 fontsize=16, fontweight='bold')

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    corr = df[num_cols].corr()

    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax, linewidths=0.5,
                cbar_kws={'shrink': 0.8})

    plt.tight_layout()
    plt.savefig(IMG_DIR / '06_correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    print("Saved: 06_correlation_heatmap.png")

# ── Plot 7: Churn by Payment Method ──────────────────────────────────────────
def plot_churn_by_payment(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Churn Rate by Payment Method',
                 fontsize=16, fontweight='bold')

    payment_churn = df.groupby('PaymentMethod')['Churn'].mean() * 100
    bars = ax.barh(payment_churn.index, payment_churn.values,
                   color=['#e74c3c' if v > 30 else '#2ecc71'
                          for v in payment_churn.values],
                   edgecolor='white')

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel('Churn Rate (%)')

    for bar, val in zip(bars, payment_churn.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(IMG_DIR / '07_churn_by_payment.png', bbox_inches='tight')
    plt.close()
    print("Saved: 07_churn_by_payment.png")

def print_key_stats(df):
    print("\n===== KEY BUSINESS INSIGHTS =====")
    print(f"Total Customers      : {len(df):,}")
    print(f"Churned Customers    : {df['Churn'].sum():,}")
    print(f"Overall Churn Rate   : {df['Churn'].mean()*100:.1f}%")
    print(f"Avg Monthly Charge   : \${df['MonthlyCharges'].mean():.2f}")
    print(f"Avg Tenure           : {df['tenure'].mean():.1f} months")
    print(f"Revenue Lost (Monthly): \${df[df['Churn']==1]['MonthlyCharges'].sum():,.0f}")

def main():
    df = load_data()
    print_key_stats(df)
    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_monthly_charges(df)
    plot_tenure_analysis(df)
    plot_churn_by_internet(df)
    plot_correlation(df)
    plot_churn_by_payment(df)
    print("\nEDA complete! All charts saved to dashboard_images/")

if __name__ == '__main__':
    main()