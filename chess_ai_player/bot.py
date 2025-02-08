import chess
import torch

from train_eval_model import *


def find_best_move(board_fen: str, eval_model):
	board = chess.Board(board_fen)
	best_move = None
	is_white_turn = board.turn
	# White wants to maximize on the evaluation, and black wants to minimize
	best_eval = -100 if is_white_turn else 100
	print(f"Current board:\n{board}")
	for move in board.legal_moves:
		board.push(move)
		possible_fen = board.fen()

		bitboard = fen_to_bitboard(possible_fen)
		bitboard_tensor = torch.from_numpy(bitboard).to(DEVICE, dtype=torch.float32)
		new_eval = float(eval_model(bitboard_tensor))
		print(f"Move: {move} give score of: {new_eval:.2f}")

		if is_white_turn and new_eval > best_eval:
			best_move = move
			best_eval = new_eval

		if not is_white_turn and new_eval < best_eval:
			best_move = move
			best_eval = new_eval

		board.pop()

	print(f"===========================================================")
	print(f"==== Best move is: {best_move}, evaluation of: {best_eval:.2f} ====")
	print(f"===========================================================")
	return best_move


if __name__ == "__main__":
	board_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/3P1B2/2N5/PPP1PPPP/R2QKBNR w KQkq - 2 4"
	eval_model = torch.load('res/model.pth')
	eval_model.eval() 
	find_best_move(board_fen, eval_model)
	board_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/3P1B2/2N5/PPP1PPPP/R2QKBNR b KQkq - 2 4"
	find_best_move(board_fen, eval_model)