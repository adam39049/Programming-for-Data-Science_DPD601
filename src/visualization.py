import matplotlib.pyplot as plt
import seaborn as sns


def plot_target_distribution(df):
    target_counts = df["Target"].value_counts()

    plt.figure(figsize=(7, 7))
    plt.pie(
        target_counts.values,
        labels=target_counts.index,
        autopct="%1.1f%%",
        colors=["#e74c3c", "#f39c12", "#27ae60"]
    )
    plt.title("Student Academic Status Distribution")
    plt.savefig("outputs/figures/01_target_distribution.png")
    plt.show()


def plot_avg_admission_grade(df):
    avg_grade = df.groupby("Target")["Admission grade"].mean().round(2)

    plt.figure(figsize=(8, 6))
    plt.bar(
        avg_grade.index,
        avg_grade.values,
        color=["#e74c3c", "#f39c12", "#27ae60"]
    )
    plt.title("Average Admission Grade by Academic Status")
    plt.xlabel("Academic Status")
    plt.ylabel("Average Admission Grade")
    plt.ylim(120, 132)

    for i, value in enumerate(avg_grade.values):
        plt.text(i, value + 0.1, str(value), ha="center", fontsize=11)

    plt.savefig("outputs/figures/02_avg_admission_grade.png")
    plt.show()


def plot_age_distribution(df):
    plt.figure(figsize=(10, 6))

    for status, color in zip(["Dropout", "Enrolled", "Graduate"],
                             ["#e74c3c", "#f39c12", "#27ae60"]):
        subset = df[df["Target"] == status]["Age at enrollment"]
        plt.hist(subset, bins=30, alpha=0.6, label=status, color=color)

    plt.title("Age at Enrollment Distribution by Academic Status")
    plt.xlabel("Age at Enrollment")
    plt.ylabel("Number of Students")
    plt.legend(title="Status")
    plt.savefig("outputs/figures/03_age_distribution.png")
    plt.show()


def plot_admission_grade_boxplot(df):
    plt.figure(figsize=(9, 6))
    sns.boxplot(
        data=df,
        x="Target",
        y="Admission grade",
        hue="Target",
        order=["Dropout", "Enrolled", "Graduate"],
        palette=["#e74c3c", "#f39c12", "#27ae60"],
        legend=False
    )
    plt.title("Admission Grade Distribution by Academic Status")
    plt.xlabel("Academic Status")
    plt.ylabel("Admission Grade")
    plt.savefig("outputs/figures/04_admission_grade_boxplot.png")
    plt.show()


def plot_semester_grades_scatter(df):
    plt.figure(figsize=(9, 7))

    for status, color in zip(["Dropout", "Enrolled", "Graduate"],
                             ["#e74c3c", "#f39c12", "#27ae60"]):
        subset = df[df["Target"] == status]
        plt.scatter(
            subset["Curricular units 1st sem (grade)"],
            subset["Curricular units 2nd sem (grade)"],
            alpha=0.3,
            label=status,
            color=color,
            s=15
        )

    plt.title("1st Semester vs 2nd Semester Grades by Academic Status")
    plt.xlabel("1st Semester Grade")
    plt.ylabel("2nd Semester Grade")
    plt.legend(title="Status")
    plt.savefig("outputs/figures/05_semester_grades_scatter.png")
    plt.show()


def plot_correlation_heatmap(df):
    selected_cols = [
        "Admission grade",
        "Age at enrollment",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (approved)",
        "Scholarship holder",
        "Debtor",
        "Tuition fees up to date",
    ]

    plt.figure(figsize=(11, 8))
    sns.heatmap(
        df[selected_cols].corr().round(2),
        annot=True,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        annot_kws={"size": 9}
    )
    plt.title("Correlation Heatmap of Key Features")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/figures/06_correlation_heatmap.png")
    plt.show()


def generate_all_visualizations(df):
    visualizations = [
        ("01 - Target Distribution", lambda: plot_target_distribution(df)),
        ("02 - Avg Admission Grade", lambda: plot_avg_admission_grade(df)),
        ("03 - Age Distribution", lambda: plot_age_distribution(df)),
        ("04 - Admission Grade Boxplot", lambda: plot_admission_grade_boxplot(df)),
        ("05 - Semester Grades Scatter", lambda: plot_semester_grades_scatter(df)),
        ("06 - Correlation Heatmap", lambda: plot_correlation_heatmap(df)),
    ]

    for name, func in visualizations:
        try:
            func()
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[ERROR] {name} - {e}")
