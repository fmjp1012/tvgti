import numpy as np
import pytest
from ..utils import elimination_matrix_h, duplication_matrix_h, elimination_matrix_hh, duplication_matrix_hh, soft_thresholding

def test_elimination_matrix_h():
    """
    テストケース: elimination_matrix_h 関数が正しく vech(S) を抽出するか確認する。
    """
    N = 3
    S = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])
    E = elimination_matrix_h(N).tocsc()
    vec_S = S.flatten(order='C')  # Row-major
    vech_S_expected = np.array([1, 2, 4, 3, 5, 6])
    vech_S_computed = E.dot(vec_S)
    np.testing.assert_array_equal(vech_S_computed.flatten(), vech_S_expected)

def test_duplication_matrix_h():
    """
    テストケース: duplication_matrix_h 関数が vech(S) から vec(S) を正しく再構築するか確認する。
    """
    N = 3
    S = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])
    D = duplication_matrix_h(N).tocsc()
    vech_S = np.array([1, 2, 4, 3, 5, 6])
    vec_S_reconstructed = D.dot(vech_S)
    S_reconstructed = vec_S_reconstructed.reshape(N, N)
    np.testing.assert_array_equal(S_reconstructed, S)

def test_elimination_matrix_hh():
    """
    テストケース: elimination_matrix_hh 関数が vechh(S) を正しく抽出するか確認する。
    """
    N = 3
    S = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])
    E_h = elimination_matrix_hh(N).tocsc()
    vec_S = S.flatten(order='C')  # Row-major
    vechh_S_expected = np.array([2, 3, 5])
    vechh_S_computed = E_h.dot(vec_S)
    np.testing.assert_array_equal(vechh_S_computed.flatten(), vechh_S_expected)

def test_duplication_matrix_hh():
    """
    テストケース: duplication_matrix_hh 関数が vechh(S) から vec(S) を正しく再構築し、
    対角成分がゼロであることを確認する。
    """
    N = 3
    S = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])
    D_h = duplication_matrix_hh(N).tocsc()
    vechh_S = np.array([2, 3, 5])
    vec_S_hollow_reconstructed = D_h.dot(vechh_S)
    S_hollow_reconstructed = vec_S_hollow_reconstructed.reshape(N, N)
    S_hollow_expected = np.array([[0, 2, 3],
                                  [2, 0, 5],
                                  [3, 5, 0]])
    np.testing.assert_array_equal(S_hollow_reconstructed, S_hollow_expected)

def test_elimination_and_duplication_h():
    """
    テストケース: elimination_matrix_h と duplication_matrix_h の連続適用が恒等変換になるか確認する。
    """
    N = 4
    S = np.random.rand(N, N)
    S = (S + S.T) / 2  # 対称行列にする
    E = elimination_matrix_h(N).tocsc()
    D = duplication_matrix_h(N).tocsc()
    vech_S = E.dot(S.flatten(order='C'))
    vec_S_reconstructed = D.dot(vech_S)
    S_reconstructed = vec_S_reconstructed.reshape(N, N)
    np.testing.assert_array_almost_equal(S_reconstructed, S)

def test_elimination_and_duplication_hh():
    """
    テストケース: elimination_matrix_hh と duplication_matrix_hh の連続適用が恒等変換になるか確認する。
    """
    N = 4
    S = np.random.rand(N, N)
    S = (S + S.T) / 2  # 対称行列にする
    np.fill_diagonal(S, 0)  # 対角成分をゼロにする
    E_h = elimination_matrix_hh(N).tocsc()
    D_h = duplication_matrix_hh(N).tocsc()
    vechh_S = E_h.dot(S.flatten(order='C'))
    vec_S_hollow_reconstructed = D_h.dot(vechh_S)
    S_hollow_reconstructed = vec_S_hollow_reconstructed.reshape(N, N)
    S_hollow_expected = S.copy()
    np.testing.assert_array_almost_equal(S_hollow_reconstructed, S_hollow_expected)

def test_large_matrix():
    """
    テストケース: 大きな行列に対しても正しく動作するか確認する。
    """
    N = 10
    S = np.random.rand(N, N)
    S = (S + S.T) / 2  # 対称行列にする

    # h-space
    E = elimination_matrix_h(N).tocsc()
    D = duplication_matrix_h(N).tocsc()
    vech_S = E.dot(S.flatten(order='C'))
    vec_S_reconstructed = D.dot(vech_S)
    S_reconstructed = vec_S_reconstructed.reshape(N, N)
    np.testing.assert_array_almost_equal(S_reconstructed, S)

    # hh-space
    S_hollow = S.copy()
    np.fill_diagonal(S_hollow, 0)
    E_h = elimination_matrix_hh(N).tocsc()
    D_h = duplication_matrix_hh(N).tocsc()
    vechh_S = E_h.dot(S_hollow.flatten(order='C'))
    vec_S_hollow_reconstructed = D_h.dot(vechh_S)
    S_hollow_reconstructed = vec_S_hollow_reconstructed.reshape(N, N)
    np.testing.assert_array_almost_equal(S_hollow_reconstructed, S_hollow)

def test_soft_thresholding_positive():
    # 正の値に対してしきい値を適用
    x = np.array([3.0, 1.0, 0.5])
    threshold = 1.0
    expected = np.array([2.0, 0.0, 0.0])
    result = soft_thresholding(x, threshold)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_soft_thresholding_negative():
    # 負の値に対してしきい値を適用
    x = np.array([-3.0, -1.0, -0.5])
    threshold = 1.0
    expected = np.array([-2.0, 0.0, 0.0])
    result = soft_thresholding(x, threshold)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_soft_thresholding_zero_threshold():
    # しきい値が0のケース
    x = np.array([3.0, -1.0, 0.0])
    threshold = 0.0
    expected = np.array([3.0, -1.0, 0.0])
    result = soft_thresholding(x, threshold)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_soft_thresholding_large_threshold():
    # しきい値がすべての値を超える場合
    x = np.array([3.0, 1.0, -2.0])
    threshold = 5.0
    expected = np.array([0.0, 0.0, 0.0])
    result = soft_thresholding(x, threshold)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


if __name__ == "__main__":
    pytest.main()
