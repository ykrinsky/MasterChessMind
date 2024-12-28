import chess
import torch

import notes


def find_best_move(board_fen: str, eval_model):
	board = chess.Board(board_fen)
	best_move = None
	best_eval = -100
	for move in board.legal_moves:
		board.push(move)
		possible_fen = board.fen()

		bitboard = notes.fen_to_bitboard(possible_fen)
		bitboard_tensor = torch.from_numpy(bitboard).to(notes.DEVICE, dtype=torch.float32)
		new_eval = eval_model(bitboard_tensor)
		print(f"Move: {move} give score of: {new_eval}")
		if new_eval > best_eval:
			best_move = move
			best_eval = new_eval

		board.pop()

	print(f"Best move is: {best_move}, and it gives eval of: {best_eval}")
	return best_move