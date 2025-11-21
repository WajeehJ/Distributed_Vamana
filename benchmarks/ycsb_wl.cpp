#include <sched.h>
#include "global.h"
#include "helper.h"
#include "workload.h"
#include "server_thread.h"
#include "table.h"
#include "row.h"
#include "index_hash.h"
#include "index_btree.h"
#include "catalog.h"
#include "manager.h"
#include "row_lock.h"
#include "query.h"
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()

#include "ycsb.h"
#include "ycsb_query.h"
#include "ycsb_store_procedure.h"

#if WORKLOAD == YCSB

int WorkloadYCSB::next_tid;

std::vector<float> base_dataset;
std::atomic<uint64_t> next_base_id{0};

std::vector<float> query_dataset;
std::atomic<uint64_t> next_query_id{0};

std::vector<std::vector<int32_t>> knn_results(100);


std::vector<float> read_fvecs(const std::string& filename, int& dimension) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }

    // Read the dimension (first 4 bytes)
    int d = 0;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));
    if (file.fail()) {
        std::cerr << "Error reading dimension." << std::endl;
        return {};
    }
    dimension = d;
    std::cout << "Vector dimension (d): " << dimension << std::endl;

    // Calculate the number of vectors
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t bytes_per_vector = sizeof(int) + dimension * sizeof(float);
    size_t num_vectors = file_size / bytes_per_vector;
    file.seekg(0, std::ios::beg); // Reset file pointer to the beginning

    std::cout << "Number of vectors: " << num_vectors << std::endl;

    std::vector<float> data;
    data.reserve(num_vectors * dimension);

    for (size_t i = 0; i < num_vectors; ++i) {
        int current_d = 0;
        file.read(reinterpret_cast<char*>(&current_d), sizeof(int));
        if (file.fail() || current_d != dimension) {
            std::cerr << "Error: Dimension mismatch or read error at vector " << i << std::endl;
            break;
        }

        // Read the vector components
        std::vector<float> vector_components(dimension);
        file.read(reinterpret_cast<char*>(vector_components.data()), dimension * sizeof(float));
        if (file.fail()) {
            std::cerr << "Error reading vector components at vector " << i << std::endl;
            break;
        }
        
        // Append to the main data vector
        data.insert(data.end(), vector_components.begin(), vector_components.end());
    }

    return data;
}

RC WorkloadYCSB::init() {
    workload::init();
    next_tid = 0;
    char * cpath = getenv("GRAPHITE_HOME");
    string path;
    if (cpath == NULL)
        path = "./benchmarks/YCSB_schema.txt";
    else {
        path = string(cpath);
        path += "/tests/apps/dbms/YCSB_schema.txt";
    }

    int dimension = 128; 
    base_dataset = read_fvecs("./benchmarks/siftsmall/siftsmall_base.fvecs", dimension);

    query_dataset = read_fvecs("./benchmarks/siftsmall/siftsmall_query.fvecs", dimension); 
    init_schema( path );
    init_table_parallel();
    return RCOK;
}

RC WorkloadYCSB::init_schema(string schema_file) {
    workload::init_schema(schema_file);
    the_table = tables[0];
    the_index = indexes[0];
    return RCOK;
}

int
WorkloadYCSB::key_to_part(uint64_t key) {
    return 0;
}

uint32_t
WorkloadYCSB::key_to_node(uint64_t key, uint32_t table_id)
{
    return key % g_num_nodes;
}

// init table in parallel
void WorkloadYCSB::init_table_parallel() {
    enable_thread_mem_pool = true;
    pthread_t p_thds[g_init_parallelism - 1];
    for (uint32_t i = 0; i < g_init_parallelism - 1; i++)
        pthread_create(&p_thds[i], NULL, threadInitTable, this);
    threadInitTable(this);

    for (uint32_t i = 0; i < g_init_parallelism - 1; i++) {
        int rc = pthread_join(p_thds[i], NULL);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }
    enable_thread_mem_pool = false;
}

void * WorkloadYCSB::init_table_slice() {
    uint32_t tid = ATOM_FETCH_ADD(next_tid, 1);
    RC rc;
    assert(tid < g_init_parallelism);
    while ((uint32_t)ATOM_FETCH_ADD(next_tid, 0) < g_init_parallelism) {}
    assert((uint32_t)ATOM_FETCH_ADD(next_tid, 0) == g_init_parallelism);

    uint64_t start = tid * g_synth_table_size / g_init_parallelism;
    uint64_t end = (tid + 1) * g_synth_table_size / g_init_parallelism;
    for (uint64_t key = start; key < end; key ++)
    {
        row_t * new_row = NULL;
        int part_id = key_to_part(key);
        rc = the_table->get_new_row(new_row, part_id);
        assert(rc == RCOK);
        // LSBs of a key indicate the node ID
        uint64_t primary_key = key * g_num_server_nodes + g_node_id;
        new_row->set_value(0, &primary_key);
        Catalog * schema = the_table->get_schema();

        for (uint32_t fid = 1; fid < schema->get_field_cnt(); fid ++) {
            char value[6] = "hello";
            new_row->set_value(fid, value);
        }


        float* vec128 = &base_dataset[next_base_id * 128];
        next_base_id.fetch_add(1);

        // int64_t randomInRange = rand() % 100;  // correct type
        // new_row->set_value(schema->get_field_cnt() - 1, &randomInRange);

        new_row->set_value(schema->get_field_cnt() - 1, vec128);


        uint64_t idx_key = primary_key;
        rc = the_index->insert(idx_key, new_row);

        assert(idx_key == new_row->get_primary_key());
        assert(rc == RCOK);
    }
    return NULL;
}

StoreProcedure *
WorkloadYCSB::create_store_procedure(TxnManager * txn, QueryBase * query)
{
    return new YCSBStoreProcedure(txn, query);
}

QueryBase *
WorkloadYCSB::gen_query()
{
    QueryBase * query = (QueryYCSB *) MALLOC(sizeof(QueryYCSB));
    new(query) QueryYCSB();
    return query;
}

QueryBase *
WorkloadYCSB::clone_query(QueryBase * query)
{
    QueryYCSB * q = (QueryYCSB *) query;
    QueryYCSB * new_q = new QueryYCSB(q->get_requests(), q->get_request_count());
    return new_q;
}

QueryBase *
WorkloadYCSB::deserialize_subquery(char * data)
{
    QueryYCSB * query = (QueryYCSB *) MALLOC(sizeof(QueryYCSB));
    new(query) QueryYCSB(data);
    return query;
}

void
WorkloadYCSB::table_to_indexes(uint32_t table_id, set<INDEX *> * indexes)
{
    assert(table_id == 0);
    indexes->insert(the_index);
}

uint64_t
WorkloadYCSB::get_primary_key(row_t * row)
{
    uint64_t key;
    row->get_value(0, &key);
    return key;
}

void
WorkloadYCSB::write_ivecs_file()
{
    std::ofstream fout("knn_out.ivecs", std::ios::binary);
    for (const auto& indices : knn_results) {
        int32_t K_int = static_cast<int32_t>(indices.size());
        fout.write(reinterpret_cast<char*>(&K_int), sizeof(int32_t));
        for (int idx : indices) {
            int32_t idx32 = static_cast<int32_t>(idx);
            fout.write(reinterpret_cast<char*>(&idx32), sizeof(int32_t));
        }
    }
    fout.close();
}


#endif
