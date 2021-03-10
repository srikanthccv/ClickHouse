(ns jepsen.nukeeper-test
  (:require [clojure.test :refer :all]
            [jepsen.nukeeper.utils :refer :all]
            [zookeeper :as zk]
            [zookeeper.data :as data]))

(defn multicreate
  [conn]
  (dorun (map (fn [v] (zk/create conn v :persistent? true)) (take 10 (zk-range)))))

(defn multidelete
  [conn]
  (dorun (map (fn [v] (zk/delete conn v)) (take 10 (zk-range)))))

(deftest a-test
  (testing "nukeeper connection"
    (let [conn (zk/connect "localhost:9181" :timeout-msec 5000)]
      (println (take 10 (zk-range)))
      (multidelete conn)
      (multicreate conn)
      (zk/create-all conn "/0")
      (zk/create conn "/0")
      (println (zk/children conn "/"))
      (zk/set-data conn "/0" (data/to-bytes "777") -1)
      (Thread/sleep 5000)
      (println "VALUE" (data/to-string (:data (zk/data conn "/0"))))
      (is (= (data/to-string (:data (zk/data conn "/0"))) "777"))
      (zk/close conn))))
