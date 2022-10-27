(ns jepsen.clickhouse.server.set
  (:require
   [clojure.tools.logging :refer :all]
   [clojure.java.jdbc :as j]
   [jepsen
    [util :as util]
    [reconnect :as rc]
    [checker :as checker]
    [client :as client]
    [generator :as gen]]
   [jepsen.clickhouse.server.client :as chc]
   [jepsen.clickhouse.utils :as chu]))

(defrecord SetClient [table-created? conn]
  client/Client
  (open! [this test node]
    (assoc this :conn (chc/client node)))

  (setup! [this test]
    (locking table-created?
      (when (compare-and-set! table-created? false true)
        (chc/with-connection [c conn]
          (j/query c "DROP TABLE IF EXISTS set")
          (j/query c "CREATE TABLE set (value Int64) Engine=MergeTree ORDER BY value")))))

  (invoke! [this test op]
    (chc/with-exception op
      (chc/with-connection [c conn]
        (util/with-retry []
          (case (:f op)
            :add (do
                    (j/query c (str "INSERT INTO set VALUES (" (:value op) ")"))
                    (assoc op :type :ok))
            :read (->> (j/query c "SELECT value FROM set")
                       (mapv :value)
                       (assoc op :type :ok, :value)))))))

  (teardown! [_ test]
    (util/timeout chc/operation-timeout
      (chc/with-connection [c conn]
        (j/query c ["DROP TABLE set"]))))

  (close! [_ test]
    (rc/close! conn)))

(defn workload
  "A generator, client, and checker for a set test."
  [opts]
  {:client    (SetClient. (atom false) nil)
   :checker   (checker/compose
                {:set (checker/set)
                 :perf (checker/perf)})
   :generator (->> (range)
                   (map (fn [x] {:type :invoke, :f :add, :value x})))
   :final-generator (gen/once {:type :invoke, :f :read, :value nil})})
