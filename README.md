# CSE 571-TeamProject
Implementation and analysis of the bi-directional search described in the paper ***"Bidirectional Search That Is Guaranteed to Meet in the Middle"*** by Robert C. Holte, Ariel Felner, Guni Sharon, and Nathan R. Sturtevant(`http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12320/12109`). This was done in the pacman environment, which was provided to us by the instructor.

## Instructions
1. Change directory to "CSE571-TeamProject"
2. Run any of the following algorithms by copying and pasting the command
   - BFS: `python pacman.py -l mediumMaze -p BFSPositionSearchAgent`
   - DFS: `python pacman.py -l mediumMaze -p DFSPositionSearchAgent`
   - Astar: `python pacman.py -l mediumMaze -p AStarPositionSearchAgent`
   - Bidirectional BFS: `python pacman.py -l mediumMaze -p biDirectionalBFSPositionSearchAgent`
   - Bidirectional Astar Search: `python pacman.py -l mediumMaze -p biDirectionalAStarPositionSearchAgent`
   - Bidirectional MM Search: `python pacman.py -l mediumMaze -p biDirectionalMMSearchAgent`
   - Bidirectional MM0 Search: `python pacman.py -l mediumMaze -p biDirectionalMM0SearchAgent`
3. The layout can be changed by altering the word after the `-l` with either `tinyMaze`, `smallMaze`, `mediumMaze`, `bigMaze`, `hugeMaze0`, `hugeMaze1`, or `hugeMaze2`. For example: `python pacman.py -l hugeMaze2 -p biDirectionalMMSearchAgent`
## Team Name
4 Sparky
## Team Members
- Kuan-Ting Shen [(@ktshen)](https://github.com/ktshen)
- Angus Liu [(@Angus-asu)](https://github.com/Angus-asu)
- Afsinur Rahman [(@arahma16)](https://github.com/arahma16)
- Mark McMillan [(@mark-mcm)](https://github.com/mark-mcm)
## Topic
Topic 1: Bi-directional Search
## Team Contribution
- AnChien Liu
• Summarized the report, and summarized the results.
- Mark McMillan
• Coding part: Implemented the MM and MM<sub>0</sub>, and performed analysis.
- Afsinur Rahman
• Coding part: Created new layouts, and performed analysis.
- Kuan-Ting Shen
• Coding part: Implemented the bidirectional adaptations of BFS and A*, and performed analysis.
Bidirectional search.
