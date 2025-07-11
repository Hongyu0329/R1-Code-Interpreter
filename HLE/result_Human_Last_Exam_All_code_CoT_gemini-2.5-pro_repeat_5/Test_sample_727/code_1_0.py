import chess
import chess.pgn
import chess.engine
import io
import sys

def find_drawing_move():
    """
    Analyzes a chess position from a specific game to find a drawing move.
    """
    # PGN of the game
    pgn_string = """
    1.d4 Nf6 2. Nf3 d5 3. g3 e6 4. Bg2 Be7 5. O-O O-O 6. b3 c5 7. dxc5 Bxc5 8. c4 dxc4 9. Qc2 Qe7 10. Nbd2 Nc6 11. Nxc4 b5 12. Nce5 Nb4 13. Qb2 Bb7 14. a3 Nc6 15. Nd3 Bb6 16. Bg5 Rfd8 17. Bxf6 gxf6 18. Rac1 Nd4 19. Nxd4 Bxd4 20. Qa2 Bxg2 21. Kxg2 Qb7+ 22. Kg1 Qe4 23. Qc2 a5 24. Rfd1 Kg7 25. Rd2 Rac8 26. Qxc8 Rxc8 27. Rxc8 Qd5 28. b4 a4 29. e3 Be5 30. h4 h5 31. Kh2 Bb2 32. Rc5 Qd6 33. Rd1 Bxa3 34. Rxb5 Qd7 35. Rc5 e5 36. Rc2 Qd5 37. Rdd2 Qb3 38. Ra2 e4 39. Nc5 Qxb4 40. Nxe4 Qb3 41. Rac2 Bf8 42. Nc5 Qb5 43. Nd3 a3 44. Nf4 Qa5 45. Ra2 Bb4 46. Rd3 Kh6 47. Rd1 Qa4 48. Rda1 Bd6 49. Kg1 Qb3 50. Ne2 Qd3 51. Nd4 Kh7 52. Kh2 Qe4 53. Rxa3 Qxh4+ 54. Kg1 Qe4 55. Ra4 Be5 56. Ne2 Qc2 57. R1a2 Qb3 58. Kg2 Qd5+ 59. f3 Qd1 60. f4 Bc7 61. Kf2 Bb6 62. Ra1 Qb3 63. Re4 Kg7 64. Re8 f5 65. Raa8 Qb4 66. Rac8 Ba5 67. Rc1 Bb6 68. Re5 Qb3 69. Re8 Qd5 70. Rcc8 Qh1 71. Rc1 Qd5 72. Rb1 Ba7 73. Re7 Bc5 74. Re5 Qd3 75. Rb7 Qc2 76. Rb5 Ba7 77. Ra5 Bb6 78. Rab5 Ba7 79. Rxf5 Qd3 80. Rxf7+ Kxf7 81. Rb7+ Kg6 82. Rxa7 Qd5 83. Ra6+ Kh7 84. Ra1 Kg6 85. Nd4 Qb7 86. Ra2 Qh1 87. Ra6+ Kf7 88. Nf3 Qb1 89. Rd6 Kg7 90. Rd5 Qa2+ 91. Rd2 Qb1 92. Re2 Qb6 93. Rc2 Qb1 94. Nd4 Qh1 95. Rc7+ Kf6 96. Rc6+ Kf7 97. Nf3 Qb1 98. Ng5+ Kg7 99. Ne6+ Kf7 100. Nd4 Qh1 101. Rc7+ Kf6 102. Nf3 Qb1 103. Rd7 Qb2+ 104. Rd2 Qb1 105. Ng1 Qb4 106. Rd1 Qb3 107. Rd6+ Kg7 108. Rd4 Qb2+ 109. Ne2 Qb1 110. e4 Qh1 111. Rd7+ Kg8 112. Rd4 Qh2+ 113. Ke3 h4 114. gxh4 Qh3+ 115. Kd2 Qxh4 116. Rd3 Kf8 117. Rf3 Qd8+ 118. Ke3 Qa5 119. Kf2 Qa7+ 120. Re3 Qd7 121. Ng3 Qd2+ 122. Kf3 Qd1+ 123. Re2 Qb3+ 124. Kg2 Qb7 125. Rd2 Qb3 126. Rd5 Ke7 127. Re5+ Kf7 128. Rf5+ Ke8 129. e5 Qa2+ 130. Kh3 Qe6
    """
    pgn = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn)
    
    # Go to the position before black's 130th move
    board = game.board()
    node = game
    # We need to go 129 full moves + 1 white move = 259 half-moves
    for _ in range(259):
        node = node.next()
    board = node.board()
    
    # List of candidate moves
    candidate_moves = {
        "A": "Qa1", "B": "Qa7", "C": "Qg2", "D": "Qf2", "E": "Qb2",
        "F": "Qd2", "G": "Qa6", "H": "Qh2", "I": "Qa8", "J": "Qa5",
        "K": "Qa4", "L": "Qc2", "M": "Qe2", "N": "Qa3"
    }

    # Find stockfish executable
    stockfish_path = None
    possible_paths = ["stockfish", "/usr/bin/stockfish", "/usr/games/stockfish", "stockfish.exe"]
    for path in possible_paths:
        try:
            with chess.engine.SimpleEngine.popen_uci(path) as engine:
                stockfish_path = path
                break
        except (FileNotFoundError, PermissionError):
            continue

    if not stockfish_path:
        print("Stockfish engine not found. Please install Stockfish and ensure it's in your system's PATH.")
        return

    best_move_label = None
    min_abs_score = float('inf')

    print(f"Analyzing position with FEN: {board.fen()}")
    print("-" * 30)

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        for label, move_san in candidate_moves.items():
            temp_board = board.copy()
            move = temp_board.parse_san(move_san)
            temp_board.push(move)
            
            # Analyze the position after the move
            info = engine.analyse(temp_board, chess.engine.Limit(time=0.5))
            score = info["score"].white().score(mate_score=10000)

            print(f"Move {label} ({move_san}): Evaluation = {score/100:.2f}")

            if abs(score) < min_abs_score:
                min_abs_score = abs(score)
                best_move_label = label

    print("-" * 30)
    print(f"The best move to secure a draw is {candidate_moves[best_move_label]}.")
    print(f"This move results in an evaluation closest to 0.00.")
    # Final answer format for the platform
    # print(f"<<<{best_move_label}>>>")


if __name__ == '__main__':
    find_drawing_move()