// tsp_ls.cpp
#include <bits/stdc++.h>
#include <filesystem>

using namespace std;

struct TSP {
    string name;
    int dimension = 0;
    vector<pair<double,double>> coords;
    vector<vector<int>> dist;

    static int iround(double x) { return (int)llround(x); }

    void loadTSPLib(const string& path) {
        ifstream in(path);
        if (!in) throw runtime_error("Cannot open file: " + path);

        string line;
        bool in_nodes = false;
        string edge_type;

        auto up = [](string x){ for (auto &c: x) c = (char)toupper((unsigned char)c); return x; };
        auto ltrim=[&](string &x){ x.erase(x.begin(), find_if(x.begin(), x.end(), [](int ch){return !isspace(ch);}));};
        auto rtrim=[&](string &x){ x.erase(find_if(x.rbegin(), x.rend(), [](int ch){return !isspace(ch);}).base(), x.end());};

        while (getline(in, line)) {
            while (!line.empty() && (line.back()=='\r' || line.back()=='\n')) line.pop_back();
            string s = line, su = up(s);

            if (!in_nodes) {
                if (su.find("DIMENSION") != string::npos) {
                    auto pos = s.find(':');
                    if (pos == string::npos) {
                        string key, tmp; istringstream iss(s); iss >> key >> tmp;
                        dimension = stoi(tmp);
                    } else {
                        string tmp = string(s.begin()+pos+1, s.end());
                        ltrim(tmp); rtrim(tmp);
                        dimension = stoi(tmp);
                    }
                } else if (su.find("EDGE_WEIGHT_TYPE") != string::npos) {
                    auto pos = s.find(':');
                    string val;
                    if (pos == string::npos) {
                        istringstream iss(s);
                        string k1,k2,tmp; iss >> k1 >> k2 >> tmp; val = tmp;
                    } else {
                        val = string(s.begin()+pos+1, s.end());
                    }
                    ltrim(val); rtrim(val);
                    if (val != "EUC_2D") throw runtime_error("Unsupported EDGE_WEIGHT_TYPE: " + val);
                } else if (su.find("NODE_COORD_SECTION") != string::npos) {
                    in_nodes = true;
                    coords.reserve(max(1, dimension));
                } else if (su == "EOF") break;
            } else {
                if (su == "EOF") break;
                if (s.empty()) continue;
                istringstream iss(s);
                int idx; double x, y;
                if (iss >> idx >> x >> y) coords.emplace_back(x, y);
            }
        }
        if ((int)coords.size() != dimension) {
            if (!coords.empty() && dimension == 0) dimension = (int)coords.size();
            else if ((int)coords.size() != dimension)
                throw runtime_error("Parsed coordinates do not match DIMENSION");
        }
        createDistanceMatrix();
    }

    void createDistanceMatrix() {
        int n = dimension;
        dist.assign(n, vector<int>(n, 0));
        for (int i = 0; i < n; ++i) {
            auto [xi, yi] = coords[i];
            for (int j = i+1; j < n; ++j) {
                auto [xj, yj] = coords[j];
                double d = sqrt((xi - xj)*(xi - xj) + (yi - yj)*(yi - yj));
                int rd = iround(d);
                dist[i][j] = dist[j][i] = rd;
            }
        }
    }

    long long pathLength(const vector<int>& perm) const {
        long long total = 0;
        for (size_t i = 0; i + 1 < perm.size(); ++i) {
            int a = perm[i] - 1, b = perm[i+1] - 1;
            total += dist[a][b];
        }
        return total;
    }
};

static vector<int> randomPermutation(int n, mt19937 &rng) {
    vector<int> p; p.reserve(n+1);
    p.push_back(1);
    vector<int> body; body.reserve(n-1);
    for (int i = 2; i <= n; ++i) body.push_back(i);
    shuffle(body.begin(), body.end(), rng);
    p.insert(p.end(), body.begin(), body.end());
    p.push_back(1);
    return p;
}

// ---------- Jump (Insertion) ----------
static pair<vector<int>, long long> jumpLocalMinimumFast(const TSP& tsp, const vector<int>& tour, int max_iters = 10000) {
    const auto &d = tsp.dist;

    vector<int> t; t.reserve(tour.size()-1);
    for (size_t i = 0; i + 1 < tour.size(); ++i) t.push_back(tour[i] - 1);
    int n = (int)t.size();

    long long cur_len = 0;
    for (int i = 0; i < n; ++i) cur_len += d[t[i]][t[(i+1)%n]];

    for (int it = 0; it < max_iters; ++it) {
        bool improved = false;

        for (int i = 0; i < n && !improved; ++i) {
            int ip = (i - 1 + n) % n, inext = (i + 1) % n;
            int a = t[ip], x = t[i], b = t[inext];

            long long remove_delta = -d[a][x] - d[x][b] + d[a][b];

            for (int j = 0; j < n; ++j) {
                if (j == i || j == ip) continue;
                int jnext = (j + 1) % n;
                int y = t[j], z = t[jnext];

                long long insert_delta = -d[y][z] + d[y][x] + d[x][z];
                long long delta = remove_delta + insert_delta;

                if (delta < 0) {
                    int city = t[i];
                    t.erase(t.begin() + i);
                    int jj = j; if (i < j) jj -= 1;
                    t.insert(t.begin() + (jj + 1), city);
                    cur_len += delta;
                    improved = true;
                    break;
                }
            }
        }
        if (!improved) break;
    }

    vector<int> result; result.reserve(n+1);
    for (int c : t) result.push_back(c + 1);
    result.push_back(result[0]);
    return {result, cur_len};
}

// ---------- Exchange (Swap) ----------
// ---------- Exchange (Swap) — fixed wrap-adjacent case ----------
static pair<vector<int>, long long> exchangeLocalMinimumFast(
    const TSP& tsp, const vector<int>& tour, int max_iters = 10000)
{
    const auto &d = tsp.dist;

    // 0-based circular tour (no duplicate end)
    vector<int> t; t.reserve(tour.size()-1);
    for (size_t k = 0; k + 1 < tour.size(); ++k) t.push_back(tour[k] - 1);
    int n = (int)t.size();

    auto tour_len = [&](const vector<int>& a)->long long {
        long long L = 0;
        for (int i = 0; i < n; ++i) L += d[a[i]][a[(i+1)%n]];
        return L;
    };

    long long cur_len = tour_len(t);

    for (int it = 0; it < max_iters; ++it) {
        bool improved = false;

        for (int i = 0; i < n && !improved; ++i) {
            int im1 = (i - 1 + n) % n;
            int ip1 = (i + 1) % n;

            for (int j = i + 1; j < n; ++j) {
                int jm1 = (j - 1 + n) % n;
                int jp1 = (j + 1) % n;

                long long delta = 0;

                if (j == ip1) {
                    // Adjacent forward: ... im1 - i - j - jp1 ...
                    delta = - (long long)d[t[im1]][t[i]] - d[t[i]][t[j]] - d[t[j]][t[jp1]]
                            + d[t[im1]][t[j]] + d[t[j]][t[i]] + d[t[i]][t[jp1]];
                } else if (i == jp1) {
                    // Wrap-adjacent: j == im1, ... jm1 - j - i - ip1 ...
                    // (j->i) cancels with (i->j), so omit both
                    delta = - (long long)d[t[jm1]][t[j]] - d[t[i]][t[ip1]]
                            + d[t[jm1]][t[i]] + d[t[j]][t[ip1]];
                } else {
                    // Non-adjacent
                    delta = - (long long)d[t[im1]][t[i]] - d[t[i]][t[ip1]]
                            - d[t[jm1]][t[j]] - d[t[j]][t[jp1]]
                            + d[t[im1]][t[j]] + d[t[j]][t[ip1]]
                            + d[t[jm1]][t[i]] + d[t[i]][t[jp1]];
                }

                if (delta < 0) {
                    swap(t[i], t[j]);
                    cur_len += delta;

                    // Optional safety check (enable in debug runs)
                    // long long check = tour_len(t);
                    // if (check != cur_len) { cur_len = check; }

                    improved = true;
                    break;
                }
            }
        }
        if (!improved) break;
    }

    vector<int> result; result.reserve(n+1);
    for (int c : t) result.push_back(c + 1);
    result.push_back(result[0]);
    return {result, cur_len};
}


// ---------- 2-Opt ----------
static pair<vector<int>, long long> twoOptLocalMinimumFast(
    const TSP& tsp, const vector<int>& tour, int max_iters = 10000)
{
    const auto &d = tsp.dist;

    vector<int> t; t.reserve(tour.size()-1);
    for (size_t i = 0; i + 1 < tour.size(); ++i) t.push_back(tour[i] - 1);
    int n = (int)t.size();

    long long cur_len = 0;
    for (int i = 0; i < n; ++i) cur_len += d[t[i]][t[(i+1)%n]];

    for (int it = 0; it < max_iters; ++it) {
        bool improved = false;

        for (int i = 0; i < n - 1 && !improved; ++i) {
            int i1 = (i + 1) % n;
            for (int j = i + 2; j < n; ++j) {
                int j1 = (j + 1) % n;
                if (i == 0 && j == n - 1) continue; // avoid breaking wrap with same edges

                long long delta = - (long long)d[t[i]][t[i1]] - d[t[j]][t[j1]]
                                  + d[t[i]][t[j]] + d[t[i1]][t[j1]];
                if (delta < 0) {
                    reverse(t.begin() + i1, t.begin() + j + 1);
                    cur_len += delta;
                    improved = true;
                    break;
                }
            }
        }
        if (!improved) break;
    }

    vector<int> result; result.reserve(n+1);
    for (int c : t) result.push_back(c + 1);
    result.push_back(result[0]);
    return {result, cur_len};
}

// ---------- Support ----------
static void ensureDir(const string& dir) {
    std::error_code ec;
    filesystem::create_directories(dir, ec);
}

// Return only the local-minimum length
static long long localSearch(const TSP& tsp, const std::string& op, std::mt19937 &rng) {
    auto current = randomPermutation(tsp.dimension, rng);

    std::vector<int> tour;
    long long len = LLONG_MAX;

    if (op == "jump") {
        std::tie(tour, len) = jumpLocalMinimumFast(tsp, current, 10000);
    } else if (op == "exchange") {
        std::tie(tour, len) = exchangeLocalMinimumFast(tsp, current, 10000);
    } else if (op == "2opt") {
        std::tie(tour, len) = twoOptLocalMinimumFast(tsp, current, 10000);
    } else {
        throw std::runtime_error("Unknown neighbourhood operator");
    }
    return len;
}

static void writeResultsAggregated(
    const std::map<std::pair<std::string,std::string>, std::pair<long long,double>>& results)
{
    ensureDir("results");
    std::ofstream ofs("results/local_search_prev.txt", std::ios::app);
    if (!ofs) {
        std::cerr << "Failed to open results/local_search_prev.txt for append\n";
        return;
    }
    for (const auto& kv : results) {
        const auto& name  = kv.first.first;
        const auto& op    = kv.first.second;
        long long minimum = kv.second.first;
        double average    = kv.second.second;

        ofs << "Name: " << name
            << "  Method: " << op
            << "   Min: " << std::fixed << std::setprecision(2) << (double)minimum
            << ", Mean: " << std::fixed << std::setprecision(2) << average
            << "\n";
    }
    std::cout << "Local search results written to results/local_search_prev.txt\n";
}

static void runAllInstances() {
    // vector<string> tsp_names = {"eil51", "eil76", "eil101", "kroA100", "kroC100", "kroD100", "lin105", "pcb442", "st70"};
    vector<string> tsp_names = {"usa13509"};
    set<string> operators = {"jump", "exchange", "2opt"}; // add "exchange", "2opt" if you want to run them too
    const int trials = 30;

    map<pair<string,string>, pair<long long,double>> results;
    mt19937 rng((uint32_t)chrono::high_resolution_clock::now().time_since_epoch().count());

    for (const auto& name : tsp_names) {
        std::cout << name << std::endl;
        try {
            string filepath = "tsp/" + name + ".tsp";
            TSP tsp;
            tsp.loadTSPLib(filepath);
            tsp.name = name;
            cout << "Running TSP instance: " << tsp.name << "\n";

            for (const auto& op : operators) {
                long long sum = 0;
                long long minimum = LLONG_MAX;

                for (int r = 0; r < trials; ++r) {
                    long long len = localSearch(tsp, op, rng);
                    sum += len;
                    if (len < minimum) minimum = len;

                    if ((r+1) % 5 == 0 || r+1 == trials) {
                        cout << "  " << op << " run " << (r+1) << "/" << trials << "\n" << flush;
                    }
                }
                double average = (double)sum / (double)trials;
                results[{name, op}] = {minimum, average};
            }
        } catch (const exception& e) {
            cerr << "Failed loading or processing " << name << ": " << e.what() << "\n";
        }
    }

    writeResultsAggregated(results);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    runAllInstances();
    return 0;
}
