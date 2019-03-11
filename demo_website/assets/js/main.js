let base_url = "http://localhost:5000/";

function init() {
  get_random_sentences();
  $("#shuffle").on("click", get_random_sentences);
  $("#getsentiment").on("click", get_sentiment_rf);
  $("#loadsentimentrf").on("click", load_sentiment_rf);
}

function get_random_sentences() {
  for (let i = 1; i < 4; i++) {
    $.ajax({
      url: base_url + "random_sentence",
      success: function (result) {
        $("#samp" + i).text(result);
      }
    });
  }
}

function get_sentiment_rf() {
  $.ajax({
    url: base_url + "random_forest/" + $("#rftext").val(),
    success: function (result) {
      $("#rfpred").text(to_sentiment(result));
    }
  })
}

function load_sentiment_rf() {
  $.ajax({
    url: base_url + "random_sentence/",
    success: function (result) {
      $("#rftext").val(result);
    }
  })
}

function to_sentiment(n) {
  switch (n) {
    case '0':
      return 'negative';
    case '1':
      return 'somewhat negative';
    case '2':
      return 'neutral';
    case '3':
      return 'somewhat positive';
    case '4':
      return 'positive';
  }
}

$(document).ready(init());