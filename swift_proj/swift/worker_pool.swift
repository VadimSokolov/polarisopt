
/**
   EMEWS loop.swift
*/

import assert;
import io;
import python;
import string;
import sys;

import EQ;
import emews;

string eval_sh = "%s/scripts/eval.sh" % getenv("EMEWS_PROJECT_ROOT");
string tmp_dir = "%s/tmp" % getenv("TURBINE_OUTPUT");

string task_code = """
import eval_wrapper

eval_wrapper.eval(r'%s', r'%s', r'%s')
r = 'OK'
""";

string parse_params_code = """
import json
# r (raw string) is required here 
payload = json.loads(r'%s')
# print(payload, flush=True)
result = '{}|{}|{}'.format(json.dumps(payload['func']), json.dumps(payload['proxies']), json.dumps(payload['parameters']))
""";

app (file out, file err) app_run_eval(string func, string proxies, string params) {
  "bash" eval_sh func proxies params @stdout=out @stderr=err;
}

(string result)run_eval(string func, string proxies, string params, string std_out_dir, int idx) {
  string out_fname = "%s/out_%d.txt" % (std_out_dir, idx);
  string err_fname = "%s/err_%d.txt" % (std_out_dir, idx);
  // printf(out_fname);
  file out <out_fname>;
  file err <err_fname>;
  (out, err) = app_run_eval(func, proxies, params) =>
  result = "OK";
}

(void v)
loop()
{
  for (boolean b = true;
       b;
       b=c)
  {
    message msg = eq_task_querier(0);
    boolean c;
    if (msg.msg_type == "status") {
      if (msg.payload == "EQ_STOP") {
        printf("loop.swift: FINAL") =>
          v = propagate() =>
          c = false;
        // finals = EQ_get();
        // printf("Swift: finals: %s", finals);
      } else {
        printf("loop.swift: got %s: exiting!", msg.payload) =>
        v = propagate() =>
        c = false;
      }
    } else {
      int eq_task_id = msg.eq_task_id;
      // payload consists of proxies and parameters
      string params_code = parse_params_code % msg.payload;
      string payload_parts[] = split(python_persist(params_code, "result"), "|");
      string params[] = parse_json_list(payload_parts[2]);
      string results[];
      foreach p,i in params
      {
        // string code = task_code % (payload_parts[0], payload_parts[1], p);
        // results[i] = python_persist(code, "r");
        results[i] = run_eval(payload_parts[0], payload_parts[1], p, tmp_dir, i);
      }
      // printf("RESULT: %s", result);
      json_result = "{\"runs\": %d}" % size(results);
      // printf("JSON RESULT: %s", json_result);
      eq_task_reporter(eq_task_id, 0, json_result) => c = true;
    }
  }

}

loop() => printf("loop.swift: normal exit.");
