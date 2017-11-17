#ifndef TEXT_UNICORN
#define TEXT_UNICORN
/*
 @brief Text Unicorn
 @date November 3rd 2017
 @author Alex Giokas
 @version 0.1.0


                             \
                              \
                               \\
                                \\
                                 >\/7
                             _.-(6'  \
                            (=___._/` \
                                 )  \ |
                                /   / |
                               /    > /
                              j    < _\
                          _.-' :      ``.
                          \ r=._\        `.
                         <`\\_  \         .`-.
                          \ r-7  `-. ._  ' .  `\
                           \`,      `-.`7  7)   )
                            \/         \|  \'  / `-._
                                       ||    .'
                                        \\  (
                                         >\  >
                                     ,.-' >.'
                                    <.'_.''
                                      <'
 */
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>
///
/// a pattern is a sentence tokenized
/// we default it to a vector of strings
/// a dense matrix and a sparse matrix aliases
///
using matrix_dense    = Eigen::MatrixXf;
using matrix_sparse   = Eigen::SparseMatrix<float>;

/// @return min-max similarity for two `pattern_type` params
template <class token_type,
          class pattern_type = std::vector<token_type>>
float min_max_sim(const pattern_type & lhs,
                  const pattern_type & rhs);
/**
 * @class vector_space
 * @brief a vector space model matrix
 * @note the template patameter `token_type` must be
 *       "searchable" inside containers for this class to
 *       function properly.
 * @note the default `pattern_type` template parameter
 *       can be changed to any stl container,
 *       provided that searching and indexing works OK
 */
template <class token_type   = std::string,
          class pattern_type = std::vector<token_type>,
          class matrix_type  = matrix_dense>
class vector_space
{
public:
    using size_type  = typename pattern_type::size_type;
    using score_type = std::pair<pattern_type,float>;

    /// @brief empty constructor
    vector_space() = default;

    /// construct with a set of unique patterns ONCE
    vector_space(std::set<pattern_type> data);

    /// @brief insert a new pattern into the VSM
    void insert(pattern_type arg);

    /// @brief return the most similar and smallest pattern indexed
    pattern_type similar_min(pattern_type arg);

    /// @brief return the most similar and largest pattern indexed
    pattern_type similar_max(pattern_type arg);

    /// @brief return any similar patterns indexed
    std::set<pattern_type> similar_any(pattern_type arg);

    /// @brief return all patterns and their respective similarity cosine
    std::deque<score_type> similar_all(pattern_type arg);

    /// @brief save all data
    void save(std::string file) const;

    /// @brief return size of VSM: columns * rows
    std::pair<size_type, size_type> size() const;

    /// @return is empty when both columns are zero
    bool is_empty() const;

    /// @return index of @param item inside @param container
    template <class item_type,
              class list_type>
    int has_index(item_type item,
                  list_type & container) const;

    /// @return the vectorized representation of pattern_type
    template <class vector_type = Eigen::VectorXf>
    vector_type vectorize(pattern_type arg) const;

	// TODO: @brief nuke this VSM
	// void clear();

protected:
    // TODO: @brief setup the weights matrix from the boolean matrix
    void set_weights_matrix();

    /// @brief print matrix on screen
    void print() const;

    /*  The actual dot product / denominator
     *      `M * V / ||M|| * ||V||` 
     *  where `M` is the matrix
     *  and `I` is the input vector 
     */
    template <class vector_type = Eigen::VectorXf>
    vector_type operator()(pattern_type arg) const;

    // filter the patterns by using the Max Coefficient of an Eigen vector
    template <class vector_type>
    std::set<pattern_type> patterns_max_coeff(vector_type arg);

    // Dense Binary Matrix (Boolean token presence)
    matrix_type matrix;
    // Dense Weighted (tf-idf) Matrix (tokens weighted - TODO)
    matrix_type weight;
    // column index (tokens), row index (patterns) 
    // value is always the matrix coeff
    std::map<token_type, unsigned int>    col_index;
    std::map<pattern_type, unsigned int>  row_index;
};
/**************************************************************
 *
 *              Template Implementations Yo!
 *
 * ***********************************************************/
template <class token_type,
          class pattern_type>
float min_max_sim(const pattern_type & lhs,
                  const pattern_type & rhs)
{
    pattern_type mine(lhs);
    pattern_type other(rhs);
    pattern_type same;
    std::sort(mine.begin(), mine.end());
    std::sort(other.begin(), other.end());
    std::set_intersection(mine.begin(), mine.end(),
                          other.begin(), other.end(),
                          std::back_inserter(same));
    float x = same.size();
    float max = std::max(mine.size(), other.size());
    return ((x - 0.f) / (max - 0.f));
}

template <class token_type,
          class pattern_type,
          class matrix_type>
vector_space<token_type,
             pattern_type,
             matrix_type
             >::vector_space(std::set<pattern_type> data)
{
    for (const auto & pattern : data) {
        insert(data);
    }
}

template <class token_type,
          class pattern_type,
          class matrix_type>    
          // TODO: partial sepcialisation for matrix_dense / sparse
          //       code below is for dense, sparse needs triplets
void vector_space<token_type,
                  pattern_type,
                  matrix_type
                  >::insert(pattern_type arg)
{
    if (has_index(arg, row_index) >= 0) {
        return;
    }
    auto last_index = row_index.size();
    row_index[arg] = last_index++;
    matrix.conservativeResize(matrix.rows() + 1, Eigen::NoChange_t());
    matrix.row(row_index[arg]).setZero();
    for (const auto & token : arg) {
        int k = has_index(token, col_index);
        if (k < 0) {
            matrix.conservativeResize(Eigen::NoChange_t(), matrix.cols() + 1);
            auto last_col = col_index.size();
            col_index[token] = last_col++;
            matrix.col(col_index[token]).setZero();
        }
        matrix(row_index[arg], col_index[token]) = 1;
    }
}

template <class token_type,
          class pattern_type,
          class matrix_type>
          // TODO: partial specialisation for dense / sparse matrix
          //       is needed
void vector_space<token_type,
                  pattern_type,
                  matrix_type
                  >::set_weights_matrix()
{
   // TODO: run ONCE?
   // TODO: iterate all col_index (patterns) #1
   //       find occurance of each token (number) and TF'it using #num / #total tokens
   //           TF = #num(token) / #num(pattern tokens)
   //
   //       find for count (col_index) = (patterns) with this Token
   //           IDF = log_e(#Patterns / #Patterns with Token)
   //
   //       final weight is = tf * idf
}

template <class token_type,
          class pattern_type,
          class matrix_type>
void vector_space<token_type,
                  pattern_type,
                  matrix_type
                  >::print() const
{
    std::cout << matrix << std::endl;  
}

template <class token_type,
          class pattern_type,
          class matrix_type>
void vector_space<token_type,
                  pattern_type,
                  matrix_type
                  >::save(std::string output) const
{
    std::string matrix_name = output;
    std::ofstream file(matrix_name);
    file << matrix << std::endl;
    file.close();
}

template <class token_type,
          class pattern_type,
          class matrix_type>
template <class item_type,
          class list_type>
int vector_space<token_type,
                 pattern_type,
                 matrix_type
                 >::has_index(item_type item, list_type & container) const
{
    auto it = container.find(item);
    if (it == container.end())
        return -1;
    return it->second; 
}

template <class token_type,
          class pattern_type,
          class matrix_type>
std::pair<typename pattern_type::size_type, 
          typename pattern_type::size_type>
    vector_space<token_type,
                 pattern_type,
                 matrix_type
                 >::size() const
{
    return std::make_pair(col_index.size(), row_index.size());
}

template <class token_type,
          class pattern_type,
          class matrix_type>
bool vector_space<token_type,
                  pattern_type,
                  matrix_type
                  >::is_empty() const
{
    return size().first == 0 && size().second == 0;
}

template <class token_type,
          class pattern_type,
          class matrix_type>
pattern_type vector_space<token_type,
                          pattern_type,
                          matrix_type
                          >::similar_min(pattern_type arg)
{
    auto res = this->operator()(arg);
    auto max_vec = patterns_max_coeff(res);
    // search for the pattern with the fewest tokens
    if (!max_vec.empty()) {
        auto min = std::min_element(max_vec.begin(), max_vec.end(), 
                        [](const pattern_type& a, const pattern_type& b) {
                                return a.size() < b.size(); });
        if (min != max_vec.end())
            return *min;
    }
    pattern_type empty{};
    return empty;
}

template <class token_type,
          class pattern_type,
          class matrix_type>
pattern_type vector_space<token_type,
                          pattern_type,
                          matrix_type
                          >::similar_max(pattern_type arg)
{
    auto res = this->operator()(arg);
    auto max_vec = patterns_max_coeff(res);
    // search for the pattern with the most tokens
    if (!max_vec.empty()) {
        auto max = std::max_element(max_vec.begin(), max_vec.end(), 
                        [](const pattern_type& a, const pattern_type& b) {
                                return a.size() < b.size(); });
        if (max != max_vec.end())
            return *max;
    }
    pattern_type empty{};
    return empty;
}

template <class token_type,
          class pattern_type,
          class matrix_type>
std::set<pattern_type> 
             vector_space<token_type,
                          pattern_type,
                          matrix_type
                          >::similar_any(pattern_type arg)
{
    auto res = this->operator()(arg);
    return patterns_max_coeff(res);
}

template <class token_type,
          class pattern_type,
          class matrix_type>
std::deque<vector_space::score_type> 
            vector_space<token_type,
                         pattern_type,
                         matrix_type
                         >::similar_all(pattern_type arg)
{
    // TODO return ALL patterns with their respective score
    //      assigned to them. No need to order or search
    //      we should assume that row index doesn't change
    //      other than increasing (???)
}

template <class token_type,
          class pattern_type,
          class matrix_type>
template <class vector_type>    // TODO verify this works for sparse vectors
                                //      else we need specialisation methods
vector_type vector_space<token_type,
                         pattern_type,
                         matrix_type
                         >::vectorize(pattern_type arg) const
{
    auto veh = vector_type(matrix.cols());
    veh.setZero();
    for (const auto & item : arg) {
        auto i = has_index(item, col_index);
        if (i >= 0) {
            veh(i) = 1.f;
        }
    }
    return veh;
}

template <class token_type,
          class pattern_type,
          class matrix_type>
template <class vector_type>    // TODO verify this works for sparse vectors
                                //      else we need specialisation methods
vector_type vector_space<token_type,
                         pattern_type,
                         matrix_type
                         >::operator()(pattern_type arg) const
{
    vector_type veh = vectorize<vector_type>(arg);
	auto den = matrix.norm() * veh.norm();
    return (matrix * veh) / den;
}

template <class token_type,
          class pattern_type,
          class matrix_type>
template <class vector_type>
std::set<pattern_type> vector_space<token_type,
                                    pattern_type,
                                    matrix_type
                                    >::patterns_max_coeff(vector_type arg)
{
    auto max = arg.maxCoeff();
    std::set<pattern_type> result {};
    for (int i = 0; i < arg.size(); ++i) {
        if (arg[i] == max) {
            auto it = std::find_if(row_index.begin(), row_index.end(),
                                   [&](const auto & rhs) { return rhs.second == i; });
            assert(it != row_index.end());
            result.insert(it->first);
        }
    }
    return result;
}
#endif
