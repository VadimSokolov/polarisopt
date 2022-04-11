
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

int TASK_TYPE = string2int(argv("task_type", "0"));


string task_code = """
import eval_DR_wrapper

eval_DR_wrapper_run(r'%s', r'%s')
r = 'OK'
""";

string parse_params_code = """
import json
# r (raw string) is required here 
payload = json.loads(r'%s')
# print(payload, flush=True)
result = '{}|{}'.format(json.dumps(payload['proxies']), json.dumps(payload['parameters']))
""";

(void v)
loop()
{
  for (boolean b = true;
       b;
       b=c)
  {
    message msg = eq_task_querier(TASK_TYPE);
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
      string params[] = parse_json_list(payload_parts[1]);
      string results[];
      foreach p,i in params
      {
        string code = task_code % (payload_parts[0], p);
        results[i] = python_persist(code, "r");
      }
      // printf("RESULT: %s", result);
      json_result = "{\"runs\": %d}" % size(results);
      // printf("JSON RESULT: %s", json_result);
      eq_task_reporter(eq_task_id, TASK_TYPE, json_result) => c = true;
    }
  }

}

loop() => printf("loop.swift: normal exit.");
