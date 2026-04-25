"""AI Chess Agent using LLM to play chess against a human opponent.

This agent uses an LLM to analyze the chess board state and make strategic
moves, providing explanations for its decisions.
"""

import os
import chess
import chess.svg
import streamlit as st
from anthropic import Anthropic
from IPython.display import display

# Initialize the Anthropic client
client = Anthropic()

# System prompt for the chess-playing AI
SYSTEM_PROMPT = """You are an expert chess player with deep knowledge of chess strategy, 
openings, tactics, and endgames. You are playing as the BLACK pieces.

When given a chess position in FEN notation, you must:
1. Analyze the current board state carefully
2. Consider strategic and tactical opportunities
3. Choose the best legal move available
4. Explain your reasoning briefly

Your response MUST follow this exact format:
MOVE: <move in UCI format, e.g., e7e5 or g8f6>
REASONING: <brief explanation of why you chose this move>

Only suggest legal moves. Think carefully before responding."""


def get_ai_move(board: chess.Board, conversation_history: list) -> tuple[str, str]:
    """Get the AI's next move using the LLM.
    
    Args:
        board: Current chess board state
        conversation_history: List of previous messages in the conversation
        
    Returns:
        Tuple of (move in UCI format, reasoning explanation)
    """
    fen = board.fen()
    legal_moves = [move.uci() for move in board.legal_moves]
    
    user_message = f"""Current board position (FEN): {fen}
    
Legal moves available: {', '.join(legal_moves)}

What is your next move?"""
    
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=conversation_history
    )
    
    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    # Parse the response
    lines = assistant_message.strip().split('\n')
    move_uci = ""
    reasoning = ""
    
    for line in lines:
        if line.startswith("MOVE:"):
            move_uci = line.replace("MOVE:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
    
    return move_uci, reasoning


def render_board(board: chess.Board) -> str:
    """Render the chess board as SVG string."""
    return chess.svg.board(board=board, size=400)


def main():
    """Main Streamlit application for the AI Chess Agent."""
    st.set_page_config(
        page_title="AI Chess Agent",
        page_icon="♟️",
        layout="centered"
    )
    
    st.title("♟️ AI Chess Agent")
    st.markdown("Play chess against Claude AI! You play as **WHITE**, AI plays as **BLACK**.")
    
    # Initialize session state
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "game_log" not in st.session_state:
        st.session_state.game_log = []
    if "status_message" not in st.session_state:
        st.session_state.status_message = "Your turn! Enter a move in UCI format (e.g., e2e4)"
    
    board = st.session_state.board
    
    # Display the chess board
    board_svg = render_board(board)
    st.components.v1.html(board_svg, height=420)
    
    # Status message
    st.info(st.session_state.status_message)
    
    # Check game over conditions
    if board.is_game_over():
        if board.is_checkmate():
            winner = "Black (AI)" if board.turn == chess.WHITE else "White (You)"
            st.success(f"🏆 Checkmate! {winner} wins!")
        elif board.is_stalemate():
            st.warning("🤝 Stalemate! It's a draw.")
        elif board.is_insufficient_material():
            st.warning("🤝 Draw due to insufficient material.")
        else:
            st.warning("🤝 The game is a draw.")
        
        if st.button("New Game"):
            st.session_state.board = chess.Board()
            st.session_state.conversation_history = []
            st.session_state.game_log = []
            st.session_state.status_message = "Your turn! Enter a move in UCI format (e.g., e2e4)"
            st.rerun()
        return
    
    # Player input (White's turn)
    if board.turn == chess.WHITE:
        col1, col2 = st.columns([3, 1])
        with col1:
            player_move = st.text_input(
                "Your move (UCI format, e.g., e2e4):",
                key="player_move_input",
                placeholder="e2e4"
            )
        with col2:
            st.write("")
            st.write("")
            submit = st.button("Make Move", type="primary")
        
        if submit and player_move:
            try:
                move = chess.Move.from_uci(player_move.strip().lower())
                if move in board.legal_moves:
                    board.push(move)
                    st.session_state.game_log.append(f"You: {player_move}")
                    st.session_state.status_message = "AI is thinking..."
                    st.rerun()
                else:
                    st.error("Illegal move! Please enter a valid move.")
            except ValueError:
                st.error("Invalid move format! Use UCI format like 'e2e4'.")
    
    # AI's turn (Black)
    elif board.turn == chess.BLACK and not board.is_game_over():
        with st.spinner("Claude is analyzing the position..."):
            ai_move_uci, reasoning = get_ai_move(
                board, 
                st.session_state.conversation_history
            )
            
            try:
                ai_move = chess.Move.from_uci(ai_move_uci)
                if ai_move in board.legal_moves:
                    board.push(ai_move)
                    st.session_state.game_log.append(f"AI: {ai_move_uci} — {reasoning}")
                    st.session_state.status_message = "Your turn! Enter a move in UCI format (e.g., e2e4)"
                    st.rerun()
            except (ValueError, chess.IllegalMoveError):
                st.error(f"AI suggested invalid move: {ai_move_uci}. Please restart.")
    
    # Game log
    if st.session_state.game_log:
        st.subheader("Game Log")
        for entry in reversed(st.session_state.game_log[-10:]):
            st.text(entry)
    
    # Reset button
    if st.button("Reset Game"):
        st.session_state.board = chess.Board()
        st.session_state.conversation_history = []
        st.session_state.game_log = []
        st.session_state.status_message = "Your turn! Enter a move in UCI format (e.g., e2e4)"
        st.rerun()


if __name__ == "__main__":
    main()
