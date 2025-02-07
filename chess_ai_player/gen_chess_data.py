import chess
import chess.pgn
import csv
import time
from stockfish import Stockfish
from collections import defaultdict

# Path to Stockfish engine
STOCKFISH_PATH = r"C:\Users\ykrin\source\repos\MasterChessMind\stockfish"
print(f"Initializing Stockfish at {STOCKFISH_PATH}")
stockfish = Stockfish(STOCKFISH_PATH)

# Optimize Stockfish settings
stockfish.set_skill_level(5)  # Lower skill level to increase speed

# Target moves count
TARGET_MOVES = 10_000_000
filtered_moves = []

# Path to Lichess PGN file
PGN_FILE_PATH = r"C:\Users\ykrin\source\repos\MasterChessMind\lichess_db_standard_rated_2014-08.pgn"
print(f"Reading PGN file: {PGN_FILE_PATH}")

# Overall evaluation distribution
overall_eval_distribution = defaultdict(int)

# Function to save results to CSV
def save_to_csv():
    csv_filename = "filtered_moves.csv"
    print(f"Saving filtered moves to {csv_filename}...")
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FEN", "Evaluation (Pawns)"])
        writer.writerows(filtered_moves)
    print(f"Saved {len(filtered_moves)} filtered moves to {csv_filename}.")

# Function to evaluate and filter moves
def process_game(game):
    global filtered_moves, overall_eval_distribution
    start_time = time.time()
    
    # Get player ratings
    white_elo = int(game.headers.get("WhiteElo", 0))
    black_elo = int(game.headers.get("BlackElo", 0))
    
    # Skip games where both players are over 1500 (searching for amateur games)
    if white_elo > 1500 and black_elo > 1500:
        print(f"Skipping game with both players over 1500 Elo (White: {white_elo}, Black: {black_elo})")
        return False
    
    board = game.board()
    move_count = 0
    eval_distribution = defaultdict(int)
    selected_moves = 0
    
    for move in game.mainline_moves():
        move_count += 1
        board.push(move)
        stockfish.set_fen_position(board.fen())
        evaluation = stockfish.get_evaluation()
        
        # Convert evaluation to pawn units
        if evaluation["type"] == "cp":
            eval_score = evaluation["value"] / 100.0  # Convert centipawns to pawns
        elif evaluation["type"] == "mate":
            continue  # Skip mate evaluations
        else:
            continue

        # Keep only moves with evaluation between 3 and 14 pawns or -3 and -14 pawns
        if 3.0 <= abs(eval_score) <= 14.0:
            filtered_moves.append((board.fen(), eval_score))
            eval_distribution[int(eval_score)] += 1
            overall_eval_distribution[int(eval_score)] += 1
            selected_moves += 1
            print(f"Move {move_count}: {board.fen()} | Eval: {eval_score} (filtered)")
        
        # Stop if we reach the target move count
        if len(filtered_moves) >= TARGET_MOVES:
            print("Target move count reached. Stopping.")
            save_to_csv()
            return True
        
        # Stop collecting moves from this game if we already selected 20
        if selected_moves >= 20:
            break
    
    elapsed_time = time.time() - start_time
    print(f"Game processed in {elapsed_time:.2f} seconds. Total filtered moves so far: {len(filtered_moves)}")
    print("Evaluation distribution for this game:")
    for score, count in sorted(eval_distribution.items()):
        print(f"  {score} pawns: {count} positions")
    
    print("Overall evaluation distribution so far:")
    for score, count in sorted(overall_eval_distribution.items()):
        print(f"  {score} pawns: {count} positions")
    
    return False

# Read PGN file
def read_pgn_file():
    start_time = time.time()
    game_count = 0
    save_interval = 1_000_000  # Save every 1 million moves
    next_save_point = save_interval
    
    with open(PGN_FILE_PATH) as pgn_file:
        while len(filtered_moves) < TARGET_MOVES:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                print("End of PGN file reached.")
                break  # End of file
            
            game_count += 1
            print(f"Processing game {game_count}...")
            
            if process_game(game):
                break
            
            # Save every 1 million moves
            if len(filtered_moves) >= next_save_point:
                save_to_csv()
                next_save_point += save_interval
    
    elapsed_time = time.time() - start_time
    print(f"Finished processing PGN file in {elapsed_time:.2f} seconds.")

# Process PGN file
read_pgn_file()

# Final save
save_to_csv()

# Print final evaluation distribution
print("Final overall evaluation distribution:")
for score, count in sorted(overall_eval_distribution.items()):
    print(f"  {score} pawns: {count} positions")
