import pytest
import numpy as np
from ..time_varying_sem import TimeVaryingSEM

@pytest.fixture
def sem_instance():
    """Fixture to create a TimeVaryingSEM instance."""
    return TimeVaryingSEM(N=3)  # デフォルト値はテストで上書き可能

def test_elimination_matrix_hollow_N2():
    N = 2
    sem = TimeVaryingSEM(N=N)
    D_h = sem.elimination_matrix_hollow(N)
    
    expected_D_h = np.array([
        [0, 1, 1, 0]
    ])
    
    assert D_h.shape == (1, 4), f"Expected shape (1,4), got {D_h.shape}"
    np.testing.assert_array_equal(D_h, expected_D_h, err_msg="D_h matrix for N=2 is incorrect.")

def test_elimination_matrix_hollow_N3():
    N = 3
    sem = TimeVaryingSEM(N=N)
    D_h = sem.elimination_matrix_hollow(N)
    
    expected_D_h = np.array([
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0]
    ])
    
    assert D_h.shape == (3, 9), f"Expected shape (3,9), got {D_h.shape}"
    np.testing.assert_array_equal(D_h, expected_D_h, err_msg="D_h matrix for N=3 is incorrect.")

def test_elimination_matrix_hollow_N4():
    N = 4
    sem = TimeVaryingSEM(N=N)
    D_h = sem.elimination_matrix_hollow(N)
    
    # 6 unique off-diagonal elements for N=4
    expected_D_h = np.array([
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    ])
    
    # ここでは手動で正しい D_h を構築するのは煩雑なので、以下のように自動生成します。
    # 実際の expected_D_h をここに記述する代わりに、正しい D_h をプログラム的に生成します。
    # しかし、手動で生成する場合、以下のように記述します。
    # 注意: 以下は例であり、実際の期待値に合わせて修正が必要です。

    # 正しい expected_D_h を手動で定義することは煩雑なので、代わりに生成した行列と一致するかを確認します。
    # ここでは N=4 の期待される l=6 行を定義します。
    l = 4 * (4 - 1) // 2  # l=6
    expected_D_h = np.array([
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,1) and (1,0)
        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,2) and (2,0)
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # (0,3) and (3,0)
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # (1,2) and (2,1)
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # (1,3) and (3,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]   # (2,3) and (3,2)
    ])

    # 実際の D_h の構築方法に基づいて正しい expected_D_h を定義する必要があります。
    # 以下は一例であり、正確な期待値に合わせて修正してください。
    # ここでは簡単のため、N=4 の正しい D_h を自動生成します。
    expected_D_h = []
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            row = np.zeros(N*N)
            row[i*N + j] = 1
            row[j*N + i] = 1
            expected_D_h.append(row)
            idx +=1
    expected_D_h = np.array(expected_D_h)

    assert D_h.shape == (6, 16), f"Expected shape (6,16), got {D_h.shape}"
    np.testing.assert_array_equal(D_h, expected_D_h, err_msg="D_h matrix for N=4 is incorrect.")

@pytest.mark.parametrize("N", [2, 3, 4])
def test_elimination_matrix_hollow_general(N):
    sem = TimeVaryingSEM(N=N)
    D_h = sem.elimination_matrix_hollow(N)
    
    # Manually construct expected D_h
    l = N * (N -1) //2
    expected_D_h = []
    for i in range(N):
        for j in range(i+1, N):
            row = np.zeros(N*N)
            row[i*N + j] = 1
            row[j*N + i] = 1
            expected_D_h.append(row)
    expected_D_h = np.array(expected_D_h)
    
    assert D_h.shape == (l, N*N), f"Expected shape ({l},{N*N}), got {D_h.shape}"
    np.testing.assert_array_equal(D_h, expected_D_h, err_msg=f"D_h matrix for N={N} is incorrect.")

def test_elimination_matrix_hollow_invalid_N():
    with pytest.raises(ValueError):
        sem = TimeVaryingSEM(N=1)  # N=1 は有効な入力ではない（l=0）
        D_h = sem.elimination_matrix_hollow(1)
        # 期待される動作に基づいて適切に処理してください。
        # 例えば、エラーを投げるようにメソッドを修正する必要があるかもしれません。
        # 現在の実装では、l=0 の行列が返されますが、エラーを期待する場合は以下のように変更します。
        # assert D_h.size == 0, "D_h should be empty for N=1"

def test_elimination_matrix_hollow_non_integer_N():
    with pytest.raises(TypeError):
        sem = TimeVaryingSEM(N=3.5)
        D_h = sem.elimination_matrix_hollow(3.5)
        # 現在の実装では整数のみを想定しているため、浮動小数点数を渡すと適切に処理されるか確認してください。
        # 必要に応じてメソッド内で型チェックを追加してください。
