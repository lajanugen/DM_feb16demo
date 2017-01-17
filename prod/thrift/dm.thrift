
namespace py demo_dm

typedef i32 int

service DM {
    void ping(),
    string run(1:string dm_input),
    void reset_ltm()
}
