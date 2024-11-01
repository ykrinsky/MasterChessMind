import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import multiprocessing as mp
import logging

import datetime
import random
import os
import sys

BITBOARD_SIZE = 773
CENTIPAWN_UNIT = 100
MAX_LEAD_SCORE = 15
HIDDEN_LAYER_SIZE = 512
BATCH_SIZE = 64
TESTSET_SIZE = 2500
EPOCH_SAVE_INTERVAL = 3 # TODO: Find something better than this mechanism

IN_GOOGLE_COLAB = True if os.getenv("COLAB_RELEASE_TAG") else False
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def fen_to_bitboard_tensor(fen: str):
    # Initialize bitboards for different piece types
    empty = 0
    white_pawns = 0
    white_knights = 0
    white_bishops = 0
    white_rooks = 0
    white_queens = 0
    white_kings = 0
    black_pawns = 0
    black_knights = 0
    black_bishops = 0
    black_rooks = 0
    black_queens = 0
    black_kings = 0

    # Split FEN into parts
    fen_board_part, fen_turn_part, fen_castling_part, *_ = fen.split()

    # Translate FEN board representation to bitboards

    # Start from the 8th rank and a-file (0-based index)
    row = 7  
    col = 0 

    for char in fen_board_part:
        if char == '/':
			# Continue to next line
            row -= 1
            col = 0
        elif char.isnumeric():
        	# Numeric means empty spaces, so we skip accordingly to the next piece in row.
            col += int(char)
        else:
        	# Convert rank and file to a square index
            square = row * 8 + col 

            if char == 'P':
                white_pawns |= 1 << square
            elif char == 'N':
                white_knights |= 1 << square
            elif char == 'B':
                white_bishops |= 1 << square
            elif char == 'R':
                white_rooks |= 1 << square
            elif char == 'Q':
                white_queens |= 1 << square
            elif char == 'K':
                white_kings |= 1 << square
            elif char == 'p':
                black_pawns |= 1 << square
            elif char == 'n':
                black_knights |= 1 << square
            elif char == 'b':
                black_bishops |= 1 << square
            elif char == 'r':
                black_rooks |= 1 << square
            elif char == 'q':
                black_queens |= 1 << square
            elif char == 'k':
                black_kings |= 1 << square

            col += 1

    pieces = [
    	white_pawns, white_knights, white_bishops, white_rooks, white_queens, white_kings, 
    	black_pawns, black_knights, black_bishops, black_rooks, black_queens, black_kings
    ]
    pieces_board_bits = []
    for piece in pieces:
    	# Pad to 64 bits for each piece and skip the '0b' prefix
    	piece_bits_str = bin(piece)[2:].zfill(64)
    	piece_bits = [int(bit) for bit in piece_bits_str]

    	pieces_board_bits.extend(piece_bits)

    # Determine player turn
    turn_bits = [1 if fen_turn_part == 'w' else 0]

    # Determine castling rights
    in_castling = lambda x: 1 if x in fen_castling_part else 0
    white_kingside_castle = in_castling('K')
    white_queenside_castle = in_castling('Q')
    black_kingside_castle = in_castling('k')
    black_queenside_castle = in_castling('q')

    castling_bits = [white_kingside_castle, white_queenside_castle, black_kingside_castle, black_queenside_castle]

    # Calculate occupancy board (and it's investion to get emptry squares) - NOT SURE IS NEEDED
    # occupancy = pawns | knights | bishops | rooks | queens | kings
    # empty = ~occupancy & 0xFFFFFFFFFFFFFFFF

    full_board_bits = pieces_board_bits + turn_bits + castling_bits
    full_board_tensor = torch.tensor(full_board_bits, dtype=torch.float32)
    #import IPython;IPython.embed()

    return full_board_tensor

if IN_GOOGLE_COLAB:
    EVALUATIONS_PATH = "/content/drive/My Drive/Colab Notebooks/res/chessData12M.csv"
    EVALUATIONS_PATH = "/content/drive/My Drive/Colab Notebooks/res/chessData100K.csv"
    FILEPATH = "/content/drive/My Drive/Colab Notebooks"
    CPU_CORES_COUNT = os.cpu_count()
else:
    EVALUATIONS_PATH = 'C:\\Users\\ykrin\\source\\repos\\chess_ai_ml\\res\\chess_eval\\tactic_evals.csv'
    EVALUATIONS_PATH = 'C:\\Users\\ykrin\\source\\repos\\chess_ai_ml\\res\\chess_eval\\chessData100K.csv'
    FILEPATH = ""
    CPU_CORES_COUNT = 3


def _process_chunk(chunk, queue):
    logging.basicConfig(level=logging.INFO, format='%(processName)s: %(message)s')
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    logging.info(f"Starting to load chunk, starting loading on {timestamp}, chunk index: {chunk.index}")
    print(f"Starting to load chunk, starting loading on {timestamp}, chunk index: {chunk.index}", flush=True)
    sys.stdout.flush()  # Explicitly flush the output buffer
    relevant_evaluations = chunk.loc[~chunk['Evaluation'].astype(str).str.contains('#')]
    all_fens = [] 
    all_evals = [] 
    for index, data in enumerate(relevant_evaluations.iloc):
        try:
            fen = data['FEN']
            #bitboard = fen_to_bitboard_tensor(fen).tolist()
            bitboard = fen_to_bitboard_tensor(fen).to(dtype=torch.bool)

            raw_eval_score = data['Evaluation']
            eval_score = int(raw_eval_score) / CENTIPAWN_UNIT
            eval_score = max(eval_score, -MAX_LEAD_SCORE)
            eval_score = min(eval_score, MAX_LEAD_SCORE)
            eval_tensor = torch.tensor([eval_score], dtype=torch.int8)
        except ValueError as e:
            logging.warning(f"Found problematic item in dataset: fen - {fen}, score - {raw_eval_score}. Skipping")
            continue

        all_fens.append(bitboard)
        all_evals.append(eval_tensor)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    logging.info(f"Finished to load chunk, finished loading on {timestamp}, chunk index: {chunk.index}")
    print(f"Finished to load chunk, finished loading on {timestamp}, chunk index: {chunk.index}", flush=True)
    sys.stdout.flush()  # Explicitly flush the output buffer

    fens_tensor = torch.stack(all_fens)
    evals_tensor = torch.stack(all_evals)

    queue.put((fens_tensor, eval_tensor))
    return

    return fens_tensor, evals_tensor

class ChessEvaluationsDataset(Dataset):
    DEFAULT_ITEM = ("r1bqk2r/pppp1ppp/2n1pn2/3P4/1b2P3/2N2Q2/PPP2PPP/R1B1KBNR b KQkq - 3 5", -0.6)

    def __init__(self, evaluations_file: str, is_for_test = False):
        queue = torch.multiprocessing.Queue()
        csv_iter = pd.read_csv(evaluations_file, chunksize=10_000)
        """
        chunk_count = 0
        for chunk in csv_iter:
            import ipdb;ipdb.set_trace()
            chunk_count += 1
        print(f"Counted {chunk_count} chunks")
        return
        """

        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        logging.info(f"Starting to load dataset. Will load all to GPU, starting loading on {timestamp}")
        print(f"P-Starting to load dataset. Will load all to GPU, starting loading on {timestamp}", flush=True)
        sys.stdout.flush()  # Explicitly flush the output buffer
        # with torch.multiprocessing.Pool(processes=4) as pool: # Maybe need to be replaced with builtin mp
            #tensor_chunks = pool.map(_process_chunk, csv_iter)

        processes = []
        num_workers = 4
        for i, chunk in enumerate(csv_iter):
            if len(processes) >= num_workers:
                processes.pop(0).join()
            
            p = mp.Process(target=_process_chunk, args=(chunk, queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        logging.info(f"Finished loading all dataset on {timestamp}")
        print(f"P-Finished to load dataset. Will load all to GPU, starting loading on {timestamp}", flush=True)

        tensors_chunks = []
        while not queue.empty():
            tensors_chunks.append(queue.get())


        #torch.cat(tensor_chunks)
        # Concatenate all chunks into a single tensor
        fens_tensors = [tensor[0] for tensor in tensor_chunks]
        evals_tensors = [tensor[1] for tensor in tensor_chunks]
        fens = torch.cat(fens_tensors, dim=0).to(DEVICE)
        evals = torch.cat(evals_tensors, dim=0).to(DEVICE)

        self.evaluations = evals
        """
        all_fens = [] 
        all_evals = [] 
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        logging.info(f"Starting to load dataset. Will load all to GPU, starting loading on {timestamp}")
        # for fen
        for index, data in enumerate(relevant_evaluations.iloc):
            if index % 200_000 == 0:
                logging.debug(f"Loaded until now, {index} positions")
            fen = data['FEN']
            #bitboard = fen_to_bitboard_tensor(fen).tolist()
            bitboard = fen_to_bitboard_tensor(fen).to(dtype=torch.bool)
            all_fens.append(bitboard)

            raw_eval_score = data['Evaluation']
            eval_score = int(raw_eval_score) / CENTIPAWN_UNIT
            eval_score = max(eval_score, -MAX_LEAD_SCORE)
            eval_score = min(eval_score, MAX_LEAD_SCORE)
            eval_tensor = torch.tensor([eval_score], dtype=torch.int8, device=DEVICE)
            all_evals.append(eval_tensor)

        self.fens_tensor = torch.stack(all_fens).to(DEVICE)
        self.evals_tensor = torch.stack(all_evals).to(DEVICE)

        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        logging.info(f"Finished loading all dataset!! Finished at {timestamp}")
        """

        #import IPython; IPython.embed()


    def __len__(self):
        return len(self.evaluations)

    def __getitem__(self, index: int):
        return self.fens_tensor[index].to(torch.float32), self.evals_tensor[index].to(torch.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(processName)s: %(message)s')
    print("Starting to run first process")
    if IN_GOOGLE_COLAB: 
        print("Running in google colab")
    else:
        print("NOT in google colab")
    print(f"Using {DEVICE} device")
    print(f"Number of available CPU cores: {CPU_CORES_COUNT}")
    # Force using spawn method for cuda (a must have when using multiple workers for loading)
    #torch.multiprocessing.set_start_method('fork', force=True)
    chess_dataset = ChessEvaluationsDataset(EVALUATIONS_PATH)

