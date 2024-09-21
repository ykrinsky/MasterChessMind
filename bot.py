import chess

import notes


def find_best_move(board_fen: str, eval_model):
	board = chess.Board(board_fen)
	best_move = None
	best_eval = -100
	for move in board.legal_moves:
		board.push(move)
		possible_fen = board.fen()

		fen_tensor = notes.fen_to_bitboard_tensor(possible_fen)
		new_eval = eval_model(fen_tensor)
		print(f"Move: {move} give score of: {new_eval}")
		if new_eval > best_eval:
			best_move = move
			best_eval = new_eval

		board.pop()

	return best_move