"""
Preprocess dataset for optimizing learning process.
"""
import pandas as pd
import matplotlib.pyplot as plt
import chess

MAX_EVAL_SCORE = 1500
CENTIPAWN_UNIT = 100


def fix_evaluation_scores(df: pd.DataFrame):
    # Deal with ending positions and specifically mates (#). Replace by maximizing the player leading, '#+1' => +1500, and '#-3' => -1500.
    df['Evaluation'] = df['Evaluation'].astype(str)
    df['Evaluation'] = df['Evaluation'].str.replace(r'#\+\d+', f'+{MAX_EVAL_SCORE}', regex=True)
    df['Evaluation'] = df['Evaluation'].str.replace(r'#-\d+', f'-{MAX_EVAL_SCORE}', regex=True)

    # Convert to numeric, setting invalid values to NaN
    df['Evaluation'] = pd.to_numeric(df['Evaluation'], errors='coerce')
    # Drop rows with NaN values
    df = df.dropna(subset=['Evaluation'])

    # Adjust score to be not under and over the max lead score to not over impact the training.
    df['Evaluation'] = df['Evaluation'].clip(lower=-MAX_EVAL_SCORE, upper=MAX_EVAL_SCORE)

    # Convert evaluation from centipawns unit to pawns unit
    df['Evaluation'] = df['Evaluation'].astype(int)/CENTIPAWN_UNIT

    df['Evaluation_int'] = df['Evaluation'].astype(int)

    return df


def draw_histogram(df: pd.DataFrame):
    # Draw how much evaluations there are in the dataset in each score 
    print(df['Evaluation_int'].value_counts())
    df['Evaluation_int'].plot(kind="hist", bins=30, title='Chess Evaluation', edgecolor='black', xlabel='Values', ylabel='Frequency')
    plt.show()


def sample_to_fix_data(df: pd.DataFrame):
    # Sample only specific amount from each column, to balance the data.
    # Letting the user to choose the column to sample, according to the data histogram presented to him.
    sampling_column = int(input("Enter column for sampling: "))
    sample_size = df['Evaluation_int'].value_counts()[sampling_column]
    print(f"Sampling {sample_size} from each column")
    fixed_df = df.groupby('Evaluation_int').apply(lambda eval_col: eval_col.sample(n=min(sample_size, len(eval_col)), random_state=42))
    fixed_df['Evaluation_int'].plot(kind="hist", bins=30, title='Fixed Chess Evaluation', edgecolor='black', xlabel='Values', ylabel='Frequency')
    plt.show()

    return fixed_df


def flip_board(fen: str, eval_score: float, eval_int: int) -> tuple[str, float, int]:
    """Flip the FEN position and reverse both Evaluation and Evaluation_int."""
    board = chess.Board(fen)
    # Flip board and negate evaluations
    return board.mirror().fen(), -eval_score, -eval_int  

def augment_with_flips(df: pd.DataFrame):
    # Filter rows where |Evaluation| is between 6 and 15
    mask = df["Evaluation"].abs().between(6, 14.9)
    # Apply flip_board to only the selected rows and append
    flipped_data = df.loc[mask, ["FEN", "Evaluation", "Evaluation_int"]].apply(
        lambda row: flip_board(*row), axis=1
    )

    df = pd.concat(
        [df, pd.DataFrame(flipped_data.tolist(), columns=["FEN", "Evaluation", "Evaluation_int"])],
        ignore_index=True
    )
    return df


def preprocess_data(evaluation_file: str):
    df = pd.read_csv(evaluation_file)
    df = fix_evaluation_scores(df)
    draw_histogram(df)
    fixed_df = sample_to_fix_data(df)
    fixed_df.to_csv(f"{evaluation_file}.fixed", index=False)
    return fixed_df

if __name__ == "__main__":
    print("Reading and fixing evaluations of chessData12M")
    df_big = fix_evaluation_scores(pd.read_csv(r"res/chessData12M.csv"))
    print("Reading and fixing evaluations of random evals")
    df_random = fix_evaluation_scores(pd.read_csv(r"res/random_evals.csv"))
    print("Reading and fixing evaluations of tactics evals")
    df_tactics = fix_evaluation_scores(pd.read_csv(r"res/tactic_evals.csv"))

    merged_df = pd.concat([df_big, df_random, df_tactics], ignore_index=True)
    print("Augmenting data with mirroring positions with a big lead")
    augmented_df = augment_with_flips(merged_df)
    import IPython; IPython.embed()
    # Manually I called sample_to_fix_data(augmented_df) and chose 5, meaning ~400K sampling.
