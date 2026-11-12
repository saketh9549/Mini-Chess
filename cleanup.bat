@echo off
cd /d "%~dp0"
del /f /q agent.py app.py arena.py cli.py eval.py eval_tuned.py game_state.py greedy_agent.py move_order.py perft.py random_agent.py run_tournaments.py search.py search_agent.py test_phase1.py test_phase2.py test_phase3.py test_phase4.py transposition_table.py tune.py tuned_agent.py 2>nul
if exist "..\semester%205" rmdir /s /q "..\semester%205" 2>nul
echo Old root files cleaned up successfully!
del "%~f0"

