/**
  * Created by dawvid on 3/15/17.
  */
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.IntParam
import org.apache.spark.sql.SQLContext
import org.apache.spark.graphx._
import org.apache.spark.graphx.util.GraphGenerators
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils


object OskiRetailers {

  case class Retailer(id: Int, storeLocType: String, testGroup: String,
                      employeePerHourPre: Int, employeePerHourPost: Int,
                      openingTimePre: Double, openingTimePost: Double,
                      closingTimePre: Double, closingTimePost: Double,
                      storeAge: Int, pop5mi: Int, comp5mi: Int)

  case class Financial(id: Int, sumPre4_5: Double, sumPost4_5: Double,
                       wk1: Double, wk2: Double, wk3: Double, wk4: Double, wk5: Double,
                       wk6: Double, wk7: Double, wk8: Double, wk9: Double, wk10: Double,
                       wk11: Double, wk12: Double, wk13: Double, wk14: Double, wk15: Double,
                       wk16: Double, wk17: Double, wk18: Double, wk19: Double, wk20: Double,
                       wk21: Double, wk22: Double, wk23: Double, wk24: Double, wk25: Double,
                       wk26: Double, wk27: Double, wk28: Double, wk29: Double, wk30: Double,
                       wk31: Double, wk32: Double, wk33: Double, wk34: Double, wk35: Double,
                       wk36: Double, wk37: Double, wk38: Double, wk39: Double, wk40: Double,
                       wk41: Double, wk42: Double, wk43: Double, wk44: Double, wk45: Double,
                       wk46: Double, wk47: Double, wk48: Double, wk49: Double, wk50: Double,
                       wk51: Double, wk52: Double)

  def main(args: Array[String]) {

    /**
      * Parsers
      */
    def parseRetailer(str: String): Retailer = {
      val line = str.split(",")
      def getTestGroup(closingTime: Double, ePH: Int): String = {
        if (closingTime == 2100.0) {
          return "Control"
        } else if (closingTime == 2000.0) {
          if (ePH == 24) return "8A"
          return "8B"
        } else {
          if (ePH == 24) return "7A"
          return "7B"
        }
      }
      Retailer(line(0).toInt, line(2) + " - " + line(1), getTestGroup(line(8).replace(":", "").toDouble, line(4).toInt), line(3).toInt, line(4).toInt, line(5).replace(":", "").toDouble, line(6).replace(":", "").toDouble, line(7).replace(":", "").toDouble, line(8).replace(":", "").toDouble, line(9).toInt, line(10).toInt * 1000, line(11).toInt)
    }

    def parseFinancial(str: String): Financial = {
      val line = str.split(",")
      var sumPre = 0.0
      var sumPost = 0.0
      for (i <- 1 to 14) {
        sumPre += line(i).toDouble
      }
      for (i <- 14 to 52) {
        sumPost += line(i).toDouble
      }
      Financial(line(0).toInt, sumPre, sumPost,
        line(1).toDouble, line(2).toDouble, line(3).toDouble, line(4).toDouble, line(5).toDouble,
        line(6).toDouble, line(7).toDouble, line(8).toDouble, line(9).toDouble, line(10).toDouble,
        line(11).toDouble, line(12).toDouble, line(13).toDouble, line(14).toDouble, line(15).toDouble,
        line(16).toDouble, line(17).toDouble, line(18).toDouble, line(19).toDouble, line(20).toDouble,
        line(21).toDouble, line(22).toDouble, line(23).toDouble, line(24).toDouble, line(25).toDouble,
        line(26).toDouble, line(27).toDouble, line(28).toDouble, line(29).toDouble, line(30).toDouble,
        line(31).toDouble, line(32).toDouble, line(33).toDouble, line(34).toDouble, line(35).toDouble,
        line(36).toDouble, line(37).toDouble, line(38).toDouble, line(39).toDouble, line(40).toDouble,
        line(41).toDouble, line(42).toDouble, line(43).toDouble, line(44).toDouble, line(45).toDouble,
        line(46).toDouble, line(47).toDouble, line(48).toDouble, line(49).toDouble, line(50).toDouble,
        line(51).toDouble, line(52).toDouble)
    }

    /**
      * Spark setup
      */
    val conf = new SparkConf().setAppName("OSKI!").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._


    /**
      * Make the Spark dataframes from APT's given data (5 sheets: 1 Retailer, 4 Financial)
      */
    val textRDD = sc.textFile("data/_APT/master.csv")
    val retailersRDD = textRDD.map(parseRetailer).cache()

    val rdd76 = sc.textFile("data/_APT/7AMto6PM.csv").map(parseFinancial).cache()
    val rdd67 = sc.textFile("data/_APT/6PMto7PM.csv").map(parseFinancial).cache()
    val rdd78 = sc.textFile("data/_APT/7PMto8PM.csv").map(parseFinancial).cache()
    val rdd89 = sc.textFile("data/_APT/8PMto9PM.csv").map(parseFinancial).cache()

    val retailersDF = retailersRDD.toDF().select("id","testGroup", "storeLocType","storeAge","pop5mi","comp5mi")
    val f76DF = rdd76.toDF()
    val f67DF = rdd67.toDF()
    val f78DF = rdd78.toDF()
    val f89DF = rdd89.toDF()
    retailersDF.registerTempTable("retailers")
    f76DF.registerTempTable("f76")

    /**
      * Create Retailers Sub-tables (i.e. subsets of the 483 retailers)
      */
    // Make 7A/B and 8A/B (Test groups) and C (Control group) classification tables
    val r7ADF = sqlContext.sql("SELECT * FROM retailers WHERE testGroup='7A'")
    val r7BDF = sqlContext.sql("SELECT * FROM retailers WHERE testGroup='7B'")
    val r8ADF = sqlContext.sql("SELECT * FROM retailers WHERE testGroup='8A'")
    val r8BDF = sqlContext.sql("SELECT * FROM retailers WHERE testGroup='7B'")
    val rCDF = sqlContext.sql("SELECT * FROM retailers WHERE testGroup='Control'")

    // By storeType and storeLocation e.g. Urban-Standalone, Rural-Stripmall
    val rMetropolitanStandalone = sqlContext.sql("SELECT * FROM retailers WHERE storeLocType='Metropolitan - Standalone'")
    val rUrbanStandalone = sqlContext.sql("SELECT * FROM retailers WHERE storeLocType='Urban - Standalone'")
    val rSuburbanStandalone = sqlContext.sql("SELECT * FROM retailers WHERE storeLocType='Suburban - Standalone'")
    val rRuralStandalone = sqlContext.sql("SELECT * FROM retailers WHERE storeLocType='Rural - Standalone'")
    val rMetropolitanStripMall = sqlContext.sql("SELECT * FROM retailers WHERE storeLocType='Metropolitan - Strip Mall'")
    val rUrbanStripMall = sqlContext.sql("SELECT * FROM retailers WHERE storeLocType='Urban - Strip Mall'")
    val rSuburbanStripMall = sqlContext.sql("SELECT * FROM retailers WHERE storeLocType='Suburban - Strip Mall'")
    val rRuralStripMall = sqlContext.sql("SELECT * FROM retailers WHERE storeLocType='Rural - Strip Mall'")

    // Write these subtables to new csv's for graph analysis
    r7ADF.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/r7A.csv")
    r7BDF.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/r7B.csv")
    r8ADF.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/r8A.csv")
    r8BDF.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/r8A.csv")
    rCDF.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rC.csv")
    rMetropolitanStandalone.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rMetroStand.csv")
    rUrbanStandalone.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rUrbanStand.csv")
    rSuburbanStandalone.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rSuburbanStand.csv")
    rRuralStandalone.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rRuralStand.csv")
    rMetropolitanStripMall.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rMetroStrip.csv")
    rUrbanStripMall.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rUrbanStrip.csv")
    rSuburbanStripMall.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rSuburbanStrip.csv")
    rRuralStripMall.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rRuralStrip.csv")


    /**
      * Create Financial tables
      */
    val rfAll76 = retailersDF.join(f76DF,"id")
    rfAll76.registerTempTable("rfAll76")
    val rfAll67 = retailersDF.join(f67DF,"id")
    rfAll67.registerTempTable("rfAll67")
    val rfAll78 = retailersDF.join(f78DF,"id")
    rfAll78.registerTempTable("rfAll78")
    val rfAll89 = retailersDF.join(f89DF,"id")
    rfAll89.registerTempTable("rfAll89")
    rfAll76.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rfAll76.csv")
    rfAll67.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rfAll67.csv")
    rfAll78.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rfAll78.csv")
    rfAll89.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rfAll89.csv")

//    val rf76 = sqlContext.sql("SELECT id, testGroup, storeLocType, storeAge, pop5mi, comp5mi, sumPre4_5 / 14 AS wkAvg76Pre, sumPost4_5 / 39 AS wkAvg76Post FROM rfAll76")
    val rf76 = sqlContext.sql("SELECT id, testGroup, sumPre4_5 / 14 AS wkAvg76Pre, sumPost4_5 / 39 AS wkAvg76Post FROM rfAll76")
    val rf67 = sqlContext.sql("SELECT id, sumPre4_5 / 14 AS wkAvg67Pre, sumPost4_5 / 39 AS wkAvg67Post FROM rfAll67")
    val rf78 = sqlContext.sql("SELECT id, sumPre4_5 / 14 AS wkAvg78Pre, sumPost4_5 / 39 AS wkAvg78Post FROM rfAll78")
    val rf89 = sqlContext.sql("SELECT id, sumPre4_5 / 14 AS wkAvg89Pre, sumPost4_5 / 39 AS wkAvg89Post FROM rfAll89")
    val rfAllAvgAll = rf76.join(rf67,"id").join(rf78,"id").join(rf89,"id")
    retailersDF.join(rfAllAvgAll, List("id","testGroup")).coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rfAllAvgAll.csv")


    /**
      * Load Emaan's compiled 7AM-9PM all Combined csv
      */
    val rfAll79 = sqlContext.read.option("header", "true").csv("data/_Compiled7AMto9PM/rfAllComb.csv")

    /**
      * Now for the fun part
      */
    // Test results individually
    rfAllAvgAll.groupBy("testGroup").avg().coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rfIndividualTestResults.csv")

    // Test results by test groups
    val rfAllPostMinusPreWkAvg = rfAllAvgAll.selectExpr("id", "testGroup", "wkAvg76Post - wkAvg76Pre AS 76Diff", "wkAvg67Post - wkAvg67Pre AS 67Diff",
      "wkAvg78Post - wkAvg78Pre AS 78Diff", "wkAvg89Post - wkAvg89Pre AS 89Diff")
    rfAllPostMinusPreWkAvg.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rfAllPostMinusPreWkAvg.csv")
    rfAllPostMinusPreWkAvg.groupBy("testGroup").avg().coalesce(1).write.mode("overwrite").option("header", "true").csv("data/rfTestGroupResults.csv")


    //find numStores, avgPop5mi, avgComp5mi, avgAge by location+type combinations
    val avgStatsByLocationAndType = sqlContext.sql("SELECT storeLocType, count(id) as numStores, avg(pop5mi) as avgPop5mi, avg(comp5mi) as avgComp5mi, avg(storeAge) as avgAge, avg(sumPre4_5) as avgTrxPre4_5, avg(sumPost4_5) as avgTrxPost4_5, avg(sumPre4_5) + avg(sumPost4_5) as avgTotalTrx FROM rfAll76 GROUP BY storeLocType ORDER BY avgPop5mi DESC")
    avgStatsByLocationAndType.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/avgStatsByLocationAndType.csv")



  }
}